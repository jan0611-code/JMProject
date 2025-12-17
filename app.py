import streamlit as st
import cv2
import numpy as np
from PIL import Image


def process_and_predict(image, min_lines, max_angle_var, min_avg_length):
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Step 3: Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # Step 4: Bilateral Filter
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=75)
    
    # Step 5: Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 17, 1
    )
    
    # Step 6: Hough Transform
    edges = cv2.Canny(thresh, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=2, maxLineGap=2)
    
    # Step 7: Feature Extraction
    num_lines = len(lines) if lines is not None else 0
    angles = []
    lengths = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lengths.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            angles.append(np.degrees(np.arctan2(y2-y1, x2-x1)) % 180)
            
    avg_len = np.mean(lengths) if lengths else 0
    angle_var = np.var(angles) if angles else 0

    # THE IF-ELSE DECISION RULES
    reasons = []
    if num_lines < min_lines:
        reasons.append(f"Garis terlalu sedikit ({num_lines})")
    if angle_var > max_angle_var:
        reasons.append(f"Sudut berantakan ({angle_var:.1f})")
    if avg_len < min_avg_length:
        reasons.append(f"Garis terlalu pendek ({avg_len:.1f})")
        
    prediction = "NORMAL" if not reasons else "ANOMALI"
    return prediction, reasons, num_lines, angle_var, avg_len, thresh

#WEB INTERFACE (by using Streamlit) 

st.set_page_config(page_title="DETEKTOR TIPE FASE LC NEMATIK")
st.title("DETEKTOR TIPE FASE ANOMALI PADA LC NEMATIK")
st.write("Silakan masukkan gambar fase untuk mendeteksi tipe fase pada LC Nematik dengan penjajaran homeotropik.")

# Sidebar for tuning
st.sidebar.header("Decision Thresholds")
m_lines = st.sidebar.slider("Minimal Kebutuhan Garis", 1, 20, 3)
m_var = st.sidebar.slider("Maksimal Varians Sudut", 100, 1000, 300)
m_len = st.sidebar.slider("Minimal Panjang Rata-rata", 1, 50, 5)

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Run Logic
    pred, reasons, n_lines, a_var, a_len, processed_img = process_and_predict(image, m_lines, m_var, m_len)
    
    # Display Result
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(processed_img, caption="Processed (Threshold)", use_container_width=True)

    if pred == "NORMAL":
        st.success(f"✅ Result: {pred}")
    else:
        st.error(f"✅ Result: {pred}")
        for r in reasons:
            st.write(f"- {r}")

    # Show Data Table
    st.table({
        "Feature": ["Garis yang terdeteksi", "Varians Sudut", "Minimal Panjang Rata-rata"],
        "Value": [n_lines, f"{a_var:.2f}", f"{a_len:.2f}"]

    })



