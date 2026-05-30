import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(
    page_title="Timber Quality Assurance Engine",
    page_icon="🪵",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #1B5E20;}
    .reason-box {background-color: #f1f3f5; padding: 20px; border-radius: 8px; font-family: monospace; white-space: pre-wrap;}
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Advanced Predictive Surface Defect Assessment Protocol")
st.write("Multi-stage spatial processing workstation for structural timber grading.")

IMAGE_DIR = "timber_images"

# --- SIDEBAR CONTROLS & DUAL INPUT PATHWAY ---
st.sidebar.header("📥 Specimen Input Stream")

# Feature 1: Upload from any source
uploaded_file = st.sidebar.file_uploader("Upload External Timber Image:", type=["png", "jpg", "jpeg"])

# Feature 2: Fallback list of baseline samples
if os.path.exists(IMAGE_DIR):
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
else:
    image_files = []

selected_baseline = None
if not uploaded_file:
    if image_files:
        selected_baseline = st.sidebar.selectbox("Or Active Calibration Queue:", options=image_files)
    else:
        st.sidebar.warning("Calibration directory empty. Please upload an external image file.")

st.sidebar.subheader("📐 Spatial Scale Factor")
ppm = st.sidebar.slider("Calibration Factor (Pixels per mm)", min_value=1.0, max_value=25.0, value=6.0, step=0.5)

st.sidebar.subheader("🎛️ Digital Image Processing Tuners")
knot_sens = st.sidebar.slider("Knot Sensitivity (Lower catches lighter knots)", min_value=30, max_value=150, value=95, step=5)
crack_sens = st.sidebar.slider("Crack Sensitivity (Lower catches finer splits)", min_value=10, max_value=100, value=40, step=5)

# --- ENGINE LOADER ---
def load_input_image():
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif selected_baseline:
        return cv2.imread(os.path.join(IMAGE_DIR, selected_baseline))
    return None

# --- MULTI-STAGE DEFECT EXTRACTION SYSTEM ---
def process_wood_physics(img, pixels_per_mm, k_thresh, c_thresh):
    h_img, w_img, _ = img.shape
    total_area_pixels = h_img * w_img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Storage arrays for localized markers
    annotated = img.copy()
    max_knot_dia = 0.0
    max_crack_len = 0.0
    wane_detected = False
    wane_extent_pct = 0.0
    
    # -------------------------------------------------------------------------
    # STAGE 1: DENSE KNOT ISOLATION (Using targeted dual-threshold mask)
    # -------------------------------------------------------------------------
    _, knot_mask = cv2.threshold(blurred, k_thresh, 255, cv2.THRESH_BINARY_INV)
    # Clean grain noise using structural opening
    k_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    knot_mask = cv2.morphologyEx(knot_mask, cv2.MORPH_OPEN, k_kernel)
    
    k_contours, _ = cv2.findContours(knot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in k_contours:
        area = cv2.contourArea(cnt)
        if area > (8 * pixels_per_mm * 8 * pixels_per_mm):  # Filter out minor artifacts
            x, y, w, h = cv2.boundingRect(cnt)
            # Ensure it is an inner defect, not edge wane cutting into the board
            if (x > 20 and y > 20 and (x+w) < (w_img-20) and (y+h) < (h_img-20)):
                dia_mm = max(w, h) / pixels_per_mm
                if dia_mm > max_knot_dia:
                    max_knot_dia = dia_mm
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, f"Knot: {dia_mm:.1f}mm", (x, y - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # -------------------------------------------------------------------------
    # STAGE 2: LINEAR CRACK/SPLIT TRACKING (Using directional gradient profile)
    # -------------------------------------------------------------------------
    # Pull directional high frequencies (Sobel vertical + horizontal derivatives)
    grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    _, crack_mask = cv2.threshold(edges, c_thresh, 255, cv2.THRESH_BINARY)
    # Re-link discontinuous linear fissures using an elongated structural element
    c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, c_kernel)
    
    c_contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in c_contours:
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        # Structural check: cracks are highly elongated features
        aspect_ratio = max(w, h) / (min(w, h) + 0.001)
        if perimeter > 40 and aspect_ratio > 3.0:
            length_mm = max(w, h) / pixels_per_mm
            if length_mm > max_crack_len:
                max_crack_len = length_mm
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 140, 0), 2)
            cv2.putText(annotated, f"Split: {length_mm:.1f}mm", (x, y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 0), 1)

    # -------------------------------------------------------------------------
    # STAGE 3: WANE DEFICIT ISOLATION (Using Convex Hull edge mapping)
    # -------------------------------------------------------------------------
    _, board_mask = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
    b_contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if b_contours:
        largest_cnt = max(b_contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_cnt)
        
        # Calculate the geometric difference between structural hull and real timber perimeter
        hull_area = cv2.contourArea(hull)
        real_area = cv2.contourArea(largest_cnt)
        deficit_area = hull_area - real_area
        
        # Track edge intrusion coordinates
        x, y, w, h = cv2.boundingRect(largest_cnt)
        edge_buffer = 30
        
        if deficit_area > (50 * pixels_per_mm * pixels_per_mm):
            # Verify if this convexity deficit touches physical boundaries
            wane_detected = True
            wane_extent_pct = (deficit_area / hull_area) * 100
            
            # Map out exactly where the geometric missing boundary sits
            cv2.drawContours(annotated, [largest_cnt], -1, (255, 0, 255), 1)
            if wane_extent_pct > 0.5:
                cv2.putText(annotated, f"Wane Region: Deficit {wane_extent_pct:.1f}%", (x + 40, y + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # -------------------------------------------------------------------------
    # STAGE 4: DETERMINISTIC WOOD TECHNOLOGY CRADING RESOLUTION MATRIX
    # -------------------------------------------------------------------------
    if max_knot_dia < 10.0 and max_crack_len == 0.0 and not wane_detected:
        assigned_grade = "Clear / Grade A"
        reason = (f"SPECIMEN COMPLIANCE: PASSED STRUCTURAL GRADE A CERTIFICATION.\n"
                  f"- Maximum Knot Diameter: {max_knot_dia:.2f} mm (Allowable tolerance: < 10 mm) -> PASSED\n"
                  f"- Maximum Crack/Split Extension: {max_crack_len:.2f} mm (Allowable tolerance: None) -> PASSED\n"
                  f"- Edge Wane Intrusion: Absolute Zero Profile Detected -> PASSED")
                  
    elif max_knot_dia <= 30.0 and max_crack_len < 50.0 and (wane_extent_pct < 4.0):
        assigned_grade = "Select / Grade B"
        wane_status = "Very Minor Profile" if wane_detected else "None"
        reason = (f"SPECIMEN COMPLIANCE: PASSED STRUCTURAL GRADE B (SELECT STRUCTURAL).\n"
                  f"- Maximum Knot Diameter: {max_knot_dia:.2f} mm (Allowable tolerance: 10 - 30 mm) -> PASSED\n"
                  f"- Maximum Crack/Split Extension: {max_crack_len:.2f} mm (Allowable tolerance: < 50 mm) -> PASSED\n"
                  f"- Edge Wane Status: {wane_status} ({wane_extent_pct:.1f}% Area Loss) -> PASSED")
                  
    else:
        assigned_grade = "Common / Grade C"
        triggers = []
        if max_knot_dia > 30.0: 
            triggers.append(f"Knot size ({max_knot_dia:.2f} mm) violates the 30 mm critical threshold boundary limit.")
        if max_crack_len >= 50.0: 
            triggers.append(f"Macro-fracture split path trajectory length ({max_crack_len:.2f} mm) exceeds the 50 mm safety cutoff.")
        if wane_extent_pct >= 4.0: 
            triggers.append(f"Significant edge wane deficit localized at {wane_extent_pct:.1f}% volumetric profile reduction.")
            
        reason = f"SPECIMEN COMPLIANCE: REJECTED FOR PREMIUM STRUCTURAL USER - DOWNGRADED TO GRADE C.\nCritical Triggers Met:\n" + "\n".join([f"  [CRITICAL FAULT] -> {t}" for t in triggers])

    extracted_metrics = {
        "max_knot": max_knot_dia,
        "max_crack": max_crack_len,
        "wane_present": "Yes" if wane_extent_pct >= 4.0 else ("Very Minor" if (0.1 < wane_extent_pct < 4.0) else "No"),
        "wane_pct": wane_extent_pct
    }
    
    return img, annotated, extracted_metrics, assigned_grade, reason

# --- PIPELINE COORDINATOR EXECUTION ---
raw_img = load_input_image()

if raw_img is not None:
    processed_img_raw, processed_img, metrics, grade, explanation = process_wood_physics(
        raw_img, ppm, knot_sens, crack_sens
    )
    
    # Layout rendering panels
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw Timber Inspection View**")
        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("**Multi-Stage Morphological Segmenter Map**")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    st.divider()
    
    # Metric Parameter Deck
    st.markdown("### 📊 Extracted Morphological Metrics (Wood Science Logging)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Assigned Quality Classification", value=grade)
    with m2:
        st.metric(label="Max Knot Diameter", value=f"{metrics['max_knot']:.1f} mm")
    with m3:
        st.metric(label="Max Crack/Split Length", value=f"{metrics['max_crack']:.1f} mm")
    with m4:
        st.metric(label="Edge Wane Profile Deviation", value=f"{metrics['wane_present']} ({metrics['wane_pct']:.1f}%)")
        
    st.markdown("### 🧠 Grading Decision Breakdown (Wood Technology Log)")
    st.markdown(f"<pre class='reason-box'>{explanation}</pre>", unsafe_allow_html=True)
else:
    st.info("System standing by. Please upload an image file from an external source or select an active specimen from the baseline queue in the sidebar controller.")