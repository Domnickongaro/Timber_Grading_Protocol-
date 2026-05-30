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
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #2E7D32;}
    .reason-box {background-color: #f1f3f5; padding: 20px; border-radius: 8px; font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Predictive Surface Defect Assessment Protocol")
st.write("Quantitative structural grading engine based on standard wood technology metrics.")

IMAGE_DIR = "timber_images"

if os.path.exists(IMAGE_DIR):
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
else:
    image_files = []

if not image_files:
    st.error("No sample images found in storage.")
    st.stop()

# --- SIDEBAR CONTROL CONTROL ---
st.sidebar.header("⚙️ Calibration Workspace")
selected_file = st.sidebar.selectbox("Select Active Specimen:", options=image_files)
image_path = os.path.join(IMAGE_DIR, selected_file)

st.sidebar.subheader("📐 Spatial Scale Factor")
# Adjust this to match your real camera setup (how many pixels span 1 mm)
pixels_per_mm = st.sidebar.slider("Calibration Factor (Pixels per mm)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

st.sidebar.subheader("🎛️ Segmentation Sensitivity")
adaptive_block = st.sidebar.slider("Adaptive Neighborhood Window", min_value=11, max_value=99, value=31, step=2)
intensity_offset = st.sidebar.slider("Contrast Offset (Sensitivity)", min_value=2, max_value=20, value=7, step=1)

# --- COMPUTER VISION EXTRACTION ENGINE ---
def analyze_timber_anomalies(path, ppm, block_size, offset):
    img = cv2.imread(path)
    if img is None:
        return None, None, {}, "Error reading specimen."
        
    h_img, w_img, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Advanced Adaptive Thresholding to extract low-contrast defects
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, block_size, offset
    )
    
    # Morphological closing to join fragmented splits/cracks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img.copy()
    
    # Metrics tracking structures
    max_knot_dia = 0.0
    max_crack_len = 0.0
    wane_detected = False
    wane_extent_pct = 0.0
    
    for cnt in contours:
        area_pixels = cv2.contourArea(cnt)
        if area_pixels < (5 * ppm * 5 * ppm): # Filter out insignificant background grain noise (<25mm²)
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Physical metric conversions
        w_mm = w / ppm
        h_mm = h / ppm
        perimeter_pixels = cv2.arcLength(cnt, True)
        
        # 1. Wane Extraction (Anomalies hugging the outer physical edge of the board)
        edge_buffer = 15 # pixels from edge
        is_touching_edge = (x < edge_buffer or y < edge_buffer or 
                             (x + w) > (w_img - edge_buffer) or 
                             (y + h) > (h_img - edge_buffer))
        
        # 2. Shape Factor Analysis (Circular vs Linear)
        # Circularity formula: 4 * pi * Area / Perimeter^2
        circularity = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
        
        # Classify based on geometric aspect ratio and position
        if is_touching_edge and circularity < 0.4:
            wane_detected = True
            wane_extent_pct = max(wane_extent_pct, (w / w_img) * 100)
            label = f"Wane: Ext {wane_extent_pct:.1f}%"
            color = (255, 0, 255) # Pink
        elif circularity < 0.25 or (max(w_mm, h_mm) / (min(w_mm, h_mm) + 0.01) > 4.0):
            # Long, thin linear shape = Check for split/crack length
            crack_length = max(w_mm, h_mm)
            if crack_length > max_crack_len:
                max_crack_len = crack_length
            label = f"Crack: {crack_length:.1f}mm"
            color = (255, 165, 0) # Orange
        else:
            # Rounder shape = Sound or Loose Knot cluster
            knot_diameter = max(w_mm, h_mm)
            if knot_diameter > max_knot_dia:
                max_knot_dia = knot_diameter
            label = f"Knot: D={knot_diameter:.1f}mm"
            color = (0, 0, 255) # Red
            
        # Draw bounding boxes and physical measurements
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- DETERMINISTIC STRUCTURAL WOOD TECHNOLOGY GRADING LOGIC ---
    if max_knot_dia < 10.0 and max_crack_len == 0.0 and not wane_detected:
        assigned_grade = "Clear / Grade A"
        reason = (f"Specimen complies fully with Grade A structural criteria. "
                  f"Maximum detected knot diameter ({max_knot_dia:.2f} mm) is safely below the 10 mm restriction. "
                  f"No linear macro-fractures (cracks) or wane anomalies were localized across the face plane.")
                  
    elif max_knot_dia <= 30.0 and max_crack_len < 50.0 and (wane_extent_pct < 8.0):
        assigned_grade = "Select / Grade B"
        reason = (f"Specimen assigned to Grade B (Select structural classification). "
                  f"Analysis observed limited macro-anomalies: Peak knot diameter is {max_knot_dia:.2f} mm "
                  f"(allowable range: 10-30 mm), maximum split trajectory extension measures {max_crack_len:.2f} mm "
                  f"(allowable constraint: <50 mm), and detected wane profiles remain minor at {wane_extent_pct:.1f}% boundary impact.")
                  
    else:
        assigned_grade = "Common / Grade C"
        triggers = []
        if max_knot_dia > 30.0: triggers.append(f"Knot size ({max_knot_dia:.2f} mm) exceeds the 30 mm critical threshold")
        if max_crack_len >= 50.0: triggers.append(f"Structural split length ({max_crack_len:.2f} mm) violates the 50 mm constraint")
        if wane_detected and wane_extent_pct >= 8.0: triggers.append(f"Significant wane profile localized at {wane_extent_pct:.1f}% extension")
        
        reason = f"Specimen downgraded to Grade C (Common/Utility grading). Disqualifying structural triggers: {'; '.join(triggers)}."

    extracted_metrics = {
        "max_knot": max_knot_dia,
        "max_crack": max_crack_len,
        "wane_present": "Yes" if wane_detected else "No",
        "wane_pct": wane_extent_pct
    }
    
    return img, annotated, extracted_metrics, assigned_grade, reason

# Execute Engine Pipeline
raw_img, processed_img, metrics, grade, explanation = analyze_timber_anomalies(
    image_path, pixels_per_mm, adaptive_block, intensity_offset
)

# --- USER INTERFACE LAYOUT ---
if raw_img is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Inspection View**")
        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("**Morphological Feature Classification Overlay**")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    st.divider()
    
    # Quantitative Dashboard
    st.markdown("### 📊 Extracted Morphological Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Assigned Quality Grade", value=grade)
    with m2:
        st.metric(label="Max Knot Diameter", value=f"{metrics['max_knot']:.1f} mm")
    with m3:
        st.metric(label="Max Crack/Split Length", value=f"{metrics['max_crack']:.1f} mm")
    with m4:
        st.metric(label="Edge Wane Profile Detected?", value=f"{metrics['wane_present']} ({metrics['wane_pct']:.1f}%)")
        
    st.markdown("### 🧠 Grading Decision Breakdown (Wood Technology Log)")
    st.markdown(f"<div class='reason-box'><strong>[RESOLUTION METRIC ENGINE LOG]:</strong> {explanation}</div>", unsafe_allow_html=True)