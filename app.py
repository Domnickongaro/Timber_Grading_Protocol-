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

uploaded_file = st.sidebar.file_uploader("Upload External Timber Image:", type=["png", "jpg", "jpeg"])

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

def load_input_image():
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif selected_baseline:
        return cv2.imread(os.path.join(IMAGE_DIR, selected_baseline))
    return None

def process_wood_physics(img, pixels_per_mm, knot_block, crack_thresh):
    h_img, w_img, _ = img.shape
    annotated = img.copy()
    
    # 1. ENHANCED PREPROCESSING (Normalize lighting via CLAHE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    
    max_knot_dia = 0.0
    max_crack_len = 0.0
    wane_detected = False
    wane_extent_pct = 0.0

    # -------------------------------------------------------------------------
    # STAGE 1: ADAPTIVE KNOT ISOLATION & CIRCULARITY FILTERING
    # -------------------------------------------------------------------------
    # Dynamic thresholding adjusts to localized dark wood spots
    knot_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, knot_block, 15
    )
    
    # Clean grain noise using structural morphology
    k_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    knot_mask = cv2.morphologyEx(knot_mask, cv2.MORPH_OPEN, k_kernel)
    
    k_contours, _ = cv2.findContours(knot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in k_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0: continue
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        
        # Knots are typically compact/circular, grain elements are elongated lines
        if area > (15 * pixels_per_mm) and circularity > 0.25:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Bound check: Ensure it's inside the timber surface, not edge artifacts
            if (20 < center[0] < w_img - 20) and (20 < center[1] < h_img - 20):
                dia_mm = (radius * 2) / pixels_per_mm
                if dia_mm > max_knot_dia:
                    max_knot_dia = dia_mm
                
                cv2.circle(annotated, center, radius, (0, 0, 255), 2)
                cv2.putText(annotated, f"Knot: {dia_mm:.1f}mm", (center[0] - radius, center[1] - radius - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # -------------------------------------------------------------------------
    # STAGE 2: ROTATED BOUNDING CRACK COMPREHENSION
    # -------------------------------------------------------------------------
    # Highlighting crisp fine fissure lines via bilateral-filtered Canny extraction
    crack_blur = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    edges = cv2.Canny(crack_blur, int(crack_thresh * 0.4), crack_thresh)
    
    # Merge tiny breaks along crack fractures
    c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, c_kernel)
    
    c_contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in c_contours:
        if cv2.contourArea(cnt) > 10:
            # Fit an oriented bounding box instead of a regular box to trace diagonal fissures
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_box, h_box), angle = rect
            box_len = max(w_box, h_box)
            box_wid = min(w_box, h_box) + 0.001
            
            if box_len > 15 and (box_len / box_wid) > 4.0:
                length_mm = box_len / pixels_per_mm
                if length_mm > max_crack_len:
                    max_crack_len = length_mm
                
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(annotated, [box], 0, (255, 140, 0), 2)
                cv2.putText(annotated, f"Split: {length_mm:.1f}mm", (int(cx), int(cy) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 0), 1)

    # -------------------------------------------------------------------------
    # STAGE 3: OTSU-DRIVEN BOARD SEGMENTATION & WANE DEFICIT
    # -------------------------------------------------------------------------
    # Automatic Otsu segmentation handles light variations across backgrounds seamlessly
    _, board_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Close any internal holes inside the wood board
    b_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, b_kernel)
    
    b_contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if b_contours:
        largest_cnt = max(b_contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_cnt)
        
        hull_area = cv2.contourArea(hull)
        real_area = cv2.contourArea(largest_cnt)
        
        if hull_area > 0:
            deficit_area = hull_area - real_area
            # Check if geometric loss breaches threshold boundaries
            if deficit_area > (30 * pixels_per_mm * pixels_per_mm):
                wane_detected = True
                wane_extent_pct = (deficit_area / hull_area) * 100
                
                cv2.drawContours(annotated, [largest_cnt], -1, (255, 0, 255), 1)
                # Draw the missing structural edge area profile
                cv2.drawContours(annotated, [hull], -1, (0, 255, 255), 1)
                
                if wane_extent_pct > 0.5:
                    cv2.putText(annotated, f"Wane Region: Deficit {wane_extent_pct:.1f}%", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # -------------------------------------------------------------------------
    # STAGE 4: DETERMINISTIC WOOD TECHNOLOGY GRADING RESOLUTION MATRIX
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw Timber Inspection View**")
        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("**Multi-Stage Morphological Segmenter Map**")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    st.divider()
    
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