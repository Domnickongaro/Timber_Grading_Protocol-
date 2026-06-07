import streamlit as st
import cv2
import numpy as np
import os
from skimage.filters import frangi

st.set_page_config(
    page_title="Timber Grading Application(Confiance V1)",
    layout="wide"
)

st.title("Timber Grading Protocol")

IMAGE_DIR = "timber_images"

# ---------------- SIDEBAR ----------------
st.sidebar.header("📥 Input")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if os.path.exists(IMAGE_DIR):
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(("png","jpg","jpeg"))])
else:
    image_files = []

selected = None
if not uploaded_file and image_files:
    selected = st.sidebar.selectbox("Sample", image_files)

ppm = st.sidebar.slider("Pixels per mm", 1.0, 25.0, 6.0)

# ---------------- LOAD IMAGE ----------------
def load_image():
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif selected:
        return cv2.imread(os.path.join(IMAGE_DIR, selected))
    return None


# ---------------- CORE PROCESSING ----------------
def process(img, ppm):

    annotated = img.copy()
    total_area = img.shape[0]*img.shape[1]

    # ---------------- STEP 1: LIGHTING NORMALIZATION ----------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ---------------- STEP 2: WOOD SEGMENTATION ----------------
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 30, 40])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img, annotated, {}, "Unknown", "Segmentation failed"

    largest = max(contours, key=cv2.contourArea)

    board_mask = np.zeros_like(mask)
    cv2.drawContours(board_mask, [largest], -1, 255, -1)

    cv2.drawContours(annotated, [largest], -1, (0,255,0), 3)

    # ---------------- STEP 3: TEXTURE ENHANCEMENT ----------------
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # ---------------- STEP 4: KNOT DETECTION ----------------
    thresh = cv2.adaptiveThreshold(
        gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,51,10
    )

    thresh = cv2.bitwise_and(thresh, board_mask)

    k_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_knot = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circ = (4*np.pi*area)/(peri**2)

        if area > (15*ppm) and circ > 0.25:
            (x,y),r = cv2.minEnclosingCircle(cnt)

            dia = (2*r)/ppm
            if dia > max_knot:
                max_knot = dia

            center = (int(x), int(y))
            cv2.circle(annotated, center, int(r), (0,0,255),2)

    # ---------------- STEP 5: CRACK DETECTION (FRANGI) ----------------
    norm_gray = gray.astype(np.float32) / 255.0

    cracks = frangi(norm_gray)

    crack_mask = (cracks > 0.15).astype(np.uint8) * 255
    crack_mask = cv2.bitwise_and(crack_mask, board_mask)

    # strengthen lines
    crack_mask = cv2.dilate(crack_mask, None, iterations=2)

    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_crack = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx,cy),(w,h),angle = rect

        length = max(w,h)
        width = min(w,h) + 0.001

        if length/width > 3:

            crack_mm = length/ppm

            if crack_mm > max_crack:
                max_crack = crack_mm

            box = cv2.boxPoints(rect)
            box = np.intp(box)

            cv2.drawContours(annotated,[box],0,(255,140,0),2)

    # ---------------- STEP 6: WANE ----------------
    hull = cv2.convexHull(largest)

    hull_area = cv2.contourArea(hull)
    real_area = cv2.contourArea(largest)

    wane_pct = 0
    wane = False

    if hull_area > 0:
        deficit = hull_area - real_area

        if deficit > (30*ppm*ppm):
            wane = True
            wane_pct = deficit/hull_area*100

            cv2.drawContours(annotated,[hull],-1,(0,255,255),2)

    # ---------------- STEP 7: GRADING ----------------
    if max_knot < 10 and max_crack < 5 and not wane:
        grade = "Grade A"
    elif max_knot <= 30 and max_crack < 50 and wane_pct < 4:
        grade = "Grade B"
    else:
        grade = "Grade C"

    metrics = {
        "knot": max_knot,
        "crack": max_crack,
        "wane": wane_pct
    }

    explanation = f"""
Knot: {max_knot:.2f} mm
Crack: {max_crack:.2f} mm
Wane: {wane_pct:.2f} %
Classification: {grade}
CONFIANCE GRADING FRAMEWORKS:
"""

    return img, annotated, metrics, grade, explanation


# ---------------- RUN ----------------
img = load_image()

if img is not None:

    raw, out, metrics, grade, exp = process(img, ppm)

    c1,c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), caption="Raw", use_container_width=True)
    with c2:
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Processed", use_container_width=True)

    st.write("---")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Grade", grade)
    m2.metric("Max Knot", f"{metrics['knot']:.1f} mm")
    m3.metric("Max Crack", f"{metrics['crack']:.1f} mm")
    m4.metric("Wane", f"{metrics['wane']:.1f}%")

    st.text(exp)

else:
    st.info("Upload an image to begin.")
