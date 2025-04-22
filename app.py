import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model_path = "best.pt"
model = YOLO(model_path)

# Page configuration
st.set_page_config(page_title="😷 Mask Detection App", layout="centered", page_icon="😷")
st.markdown("<h1 style='text-align: center;'>😷 Face Mask Detection System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.markdown("## 🛠️ Settings")
    mode = st.radio("🎯 Choose Mode", ["Test Image (Upload / Capture)"])
    st.markdown("---")
    st.markdown("Made with ❤️ by **Prabh**")

# Test Image Mode
if mode == "Test Image (Upload / Capture)":
    st.subheader("📸 Test with Image")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(frame, verbose=False)
        annotated = results[0].plot()

        # Accuracy breakdown
        mask_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)
        no_mask_count = sum(1 for c in results[0].boxes.cls if int(c) == 1)
        total = mask_count + no_mask_count
        if total > 0:
            accuracy = (mask_count / total) * 100
            st.success(f"✅ Masked: {mask_count}, ❌ No Mask: {no_mask_count}, 🎯 Accuracy: {accuracy:.2f}%")
        else:
            st.info("No faces detected.")

        rgb_result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_result, channels="RGB", caption="🧠 Detection Result")
