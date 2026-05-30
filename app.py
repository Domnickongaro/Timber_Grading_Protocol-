import streamlit as st
import cv2
import numpy as np
import os
import math

st.set_page_config(
    page_title="Timber Quality Assurance Engine",
    page_icon="🪵",
    layout="wide"
)

# Custom styling for UI elements
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
knot_sens = st.sidebar.slider("Knot Adaptive Block Size", min_value=11, max_value=151, value=51, step=2)
crack_sens = st.sidebar.slider("Crack Edge Upper Threshold", min_value=30, max_value=200, value=100, step=5)

# --- ENGINE LOADER ---
def load_input_image():
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif selected_baseline:
        return cv2.imread(os.path.join(IMAGE_DIR, selected_baseline))
    return None

# --- MULTI-STAGE DEFECT EXTRACTION SYSTEM ---
def process_wood_physics(img, pixels_per_mm, knot_block, crack_thresh):
    h_img, w_img, _ = img.shape
    total_area_pixels = h_img * w_img
    annotated = img.copy()
    
    # -------------------------------------------------------------------------
    # STAGE 0: BACKGROUND REMOVAL & TIMBER ISOLATION (FIXES DESK DETECTION ERRORS)
    # -------------------------------------------------------------------------
    # Gating the image via HSV to focus only on wood tones
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_wood = np.array([5, 30, 40])
    upper_wood = np.array([30, 255, 235])
    wood_color_mask = cv2.inRange(hsv, lower_wood, upper_wood)
    
    # Close internal grain patterns to achieve a solid silhouette
    b_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    refined_mask = cv2.morphologyEx(wood_color_mask, cv2.MORPH_CLOSE, b_kernel)
    
    b_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fallback back into Otsu segmentation if the color profile fails entirely
    if not b_contours or cv2.contourArea(max(b_contours, key=cv2.contourArea)) < (total_area_pixels * 0.05):
        gray_fallback = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, refined_mask = cv2.threshold(gray_fallback, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_knot_dia = 0.0
    max_crack_len = 0.0
    wane_detected = False
    wane_extent_pct = 0.0
    
    if not b_contours:
        return img, annotated, {"max_knot": 0, "max_crack": 0, "wane_present": "No", "wane_pct": 0}, "Unknown", "CRITICAL ERROR: Timber surface could not be segmented from background workspace environment."

    # Identify the ultimate timber chunk boundary
    largest_cnt = max(b_contours, key=cv2.contourArea)
    
    # Create an absolute exclusion matrix (strict mask)
    strict_board_mask = np.zeros_like(refined_mask)
    cv2.drawContours(strict_board_mask, [largest_cnt], -1, 255, -1)
    
    # Trace a solid green indicator line around the board tracking profile
    cv2.drawContours(annotated, [largest_cnt], -1, (0, 255, 0), 3) 

    # Equalize contrast profiles internally using CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # -------------------------------------------------------------------------
    # STAGE 1: ADAPTIVE KNOT ISOLATION (BOUNDED INSIDE WOOD MASK)
    # -------------------------------------------------------------------------
    knot_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, knot_block, 12
    )
    # Wipe away all detections happening on the surrounding desk/background
    knot_mask = cv2.bitwise_and(knot_mask, strict_board_mask)
    
    k_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    knot_mask = cv2.morphologyEx(knot_mask, cv2.MORPH_OPEN, k_kernel)
    
    k_contours, _ = cv2.findContours(knot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in k_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Suppress long linear grain noise, highlight structural circular knots
        if area > (12 * pixels_per_mm) and circularity > 0.20:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Ensure center points exist inside the live timber zone
            if cv2.pointPolygonTest(largest_cnt, (x, y), False) >= 0:
                dia_mm = (radius * 2) / pixels_per_mm
                if dia_mm > max_knot_dia:
                    max_knot_dia = dia_mm
                
                cv2.circle(annotated, center, radius, (0, 0, 255), 2)
                cv2.putText(annotated, f"Knot: {dia_mm:.1f}mm", (center[0] - radius, center[1] - radius - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # -------------------------------------------------------------------------
    # STAGE 2: ROTATED MIN-AREA CRACK TRACKING (BOUNDED INSIDE WOOD MASK)
    # -------------------------------------------------------------------------
    crack_blur = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    edges = cv2.Canny(crack_blur, int(crack_thresh * 0.4), crack_thresh)
    # Wipe away external desk edge shadows
    edges = cv2.bitwise_and(edges, strict_board_mask)
    
    c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, c_kernel)
    
    c_contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in c_contours:
        if cv2.contourArea(cnt) > 8:
            # Capture the crack length diagonally using minimum bounding boxes
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_box, h_box), angle = rect
            box_len = max(w_box, h_box)
            box_wid = min(w_box, h_box) + 0.001
            
            # Elongation aspect verification
            if box_len > 12 and (box_len / box_wid) > 3.5:
                length_mm = box_len / pixels_per_mm
                if length_mm > max_crack_len:
                    max_crack_len = length_mm
                
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(annotated, [box], 0, (255, 140, 0), 2)

    # -------------------------------------------------------------------------
    # STAGE 3: CONVEX HULL WANE ASSESSMENT
    # -------------------------------------------------------------------------
    hull = cv2.convexHull(largest_cnt)
    hull_area = cv2.contourArea(hull)
    real_area = cv2.contourArea(largest_cnt)
    
    if hull_area > 0:
        deficit_area = hull_area - real_area
        if deficit_area > (40 * pixels_per_mm * pixels_per_mm):
            wane_detected = True
            wane_extent_pct = (deficit_area / hull_area) * 100
            
            # Render expected structural shape overlay profile line in yellow
            cv2.drawContours(annotated, [hull], -1, (0, 255, 255), 2)
            if wane_extent_pct > 1.0:
                cv2.putText(annotated, f"Wane Region: Deficit {wane_extent_pct:.1f}%", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # -------------------------------------------------------------------------
    # STAGE 4: DETERMINISTIC TIMBER GRADING RESOLUTION MATRIX
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
        "wane_present": "Yes" if wane_extent_pct >= 4.0 else ("Minor" if wane_extent_pct > 0.1 else "No"),
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
    st.markdown("### 📊 Extracted Morphological Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Assigned Quality Classification", value=grade)
    with m2:
        st.metric(label="Max Knot Diameter", value=f"{metrics['max_knot']:.1f} mm")
    with m3:
        st.metric(label="Max Crack/Split Length", value=f"{metrics['max_crack']:.1f} mm")
    with m4:
        st.metric(label="Edge Wane Profile Deviation", value=f"{metrics['wane_present']} ({metrics['wane_pct']:.1f}%)")
        
    st.markdown("### Grading Decision Breakdown ")
    st.markdown(f"<pre class='reason-box'>{explanation}</pre>", unsafe_allow_html=True)
else:
    st.info("System standing by. Please upload an image file from an external source or select an active specimen from the baseline queue in the sidebar controller.")