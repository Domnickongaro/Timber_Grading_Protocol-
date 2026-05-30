import streamlit as st
import cv2
import numpy as np
import os

st.set_page_config(
    page_title="Timber Quality Assurance Engine",
    page_icon="🪵",
    layout="wide"
)

# Custom styling for a clean engineering interface
st.markdown("""
    <style>
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    .reason-box {background-color: #f1f3f5; padding: 20px; border-radius: 8px; font-style: italic;}
    </style>
""", unsafe_allow_html=True)

st.title("🪵 Predictive Surface Defect Assessment Protocol")
st.write("Automated visual inspection framework for quality assurance and structural timber grading.")

# --- SIDEBAR CONTROL WORKSTATION ---
st.sidebar.header("📋 Workstation Controls")

IMAGE_DIR = "timber_images"

# Load and sort the 24 sample images
if os.path.exists(IMAGE_DIR):
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
else:
    image_files = []

if not image_files:
    st.sidebar.error(f"No sample images found in '{IMAGE_DIR}'. Please check your repository configuration.")
    st.stop()

# Dropdown selector to pick exactly 1 timber sample at a time
selected_file = st.sidebar.selectbox(
    "Queue Active Board Select:",
    options=image_files,
    index=0
)
image_path = os.path.join(IMAGE_DIR, selected_file)

# Dynamic tuning controls for the computer vision logic
st.sidebar.subheader("🔧 Spatiotemporal Calibration")
intensity_offset = st.sidebar.slider(
    "Contrast Sensitivity (Adaptive Threshold)", 
    min_value=3, max_value=25, value=11, step=2,
    help="Higher values isolate severe defects; lower values capture subtle grain variations."
)
min_defect_area = st.sidebar.slider(
    "Minimum Defect Area (Pixels)", 
    min_value=50, max_value=5000, value=300, step=50,
    help="Filter threshold to separate negligible wood grain texture from actual defects."
)

# --- IMAGE PROCESSING AND FEATURE EXTRACTION PIPELINE ---
@st.cache_data(show_spinner=False)
def process_timber_face(path, offset, min_area):
    # Read image in raw BGR format
    img = cv2.imread(path)
    if img is None:
        return None, None, 0, 0, "Error loading image file."

    # 1. Image Preprocessing & Spatial Smoothing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Adaptive Binarization (Handles uneven lighting across the 24 samples)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, offset
    )

    # 3. Geometric Feature Extraction
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated_img = img.copy()
    defect_count = 0
    max_defect_size = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Apply the filter size constraint
        if area > min_area:
            defect_count += 1
            if area > max_defect_size:
                max_defect_size = area

            # Extract bounding boxes for localization metrics
            x, y, w, h = cv2.boundingRect(cnt)

            # Label detected nodes/defects visually
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                annotated_img, f"ID:{defect_count} ({int(area)}px)", 
                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
            )

    # 4. Deterministic Rule-Based Grading Engine
    if defect_count == 0:
        grade = "Grade A (Premium / Clear)"
        reason = "The structural assessment detected zero surface anomalies or density deviations exceeding the minimum tolerance threshold."
    elif max_defect_size < 1500:
        grade = "Grade B (Select / Structural)"
        reason = f"Assigned to Grade B due to minor structural anomalies. System extracted {defect_count} anomalous feature(s), with the peak localized defect area measuring {int(max_defect_size)} square pixels—remaining within allowable thresholds for load-bearing timber."
    else:
        grade = "Grade C (Utility / Common)"
        reason = f"Downgraded to Grade C due to major localized structural deviations. Feature ID verification caught a severe anomaly spanning {int(max_defect_size)} square pixels, violating the structural integrity tolerances required for premium certification."

    return img, annotated_img, defect_count, max_defect_size, grade, reason

# Execute the processing pipeline
raw_img, processed_img, count, max_size, quality_grade, explanation = process_timber_face(
    image_path, intensity_offset, min_defect_area
)

# --- USER INTERFACE DISPLAY LAYOUT ---
if raw_img is not None:
    st.subheader(f"Active Inspection Unit: `{selected_file}`")

    # Side-by-side comparative inspection display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Timber Input View**")
        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("**Automated Feature Segmentation Overlay**")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.divider()

    # Metrics Panel
    st.markdown("### 📊 Automated Inspection Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.metric(label="Assigned Quality Classification", value=quality_grade.split(" (")[0])
    with m_col2:
        st.metric(label="Total Distinct Surface Defects", value=f"{count} Detected")
    with m_col3:
        st.metric(label="Critical Defect Footprint", value=f"{int(max_size)} px²")

    # Decision Reasoning Breakdown Output
    st.markdown("### 🧠 Grading Decision Breakdown")
    st.markdown(f"<div class='reason-box'><strong>System Evaluation Note:</strong> {explanation}</div>", unsafe_allow_html=True)

else:
    st.error("Error running processing protocol on selected image stream.")
