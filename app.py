import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# Check if numpy and torch are available
assert np.__version__, "Numpy is not available."
assert torch.__version__, "Torch is not available."

# Load YOLOv8 model
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
    show_about = st.checkbox("📖 About", value=False)

# About section
if show_about:
    st.markdown("## 📖 About")
    st.markdown(
        """
        **Face Mask Detection System** identifies whether people are wearing masks using deep learning.

        **Technology Stack:**
        - **Model:** YOLOv8 (Ultralytics)
        - **Framework:** Streamlit
        - **Libraries:** OpenCV, NumPy, PIL, PyTorch

        **Workflow:**
        1. Upload or capture an image.
        2. The YOLOv8 model detects faces and classifies them as "Masked" or "No Mask".
        3. Results are displayed with bounding boxes.
        """
    )
    st.markdown("---")

# Main Image Testing
if mode == "Test Image (Upload / Capture)":
    st.subheader("📸 Test with Image")

    uploaded = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        frame = np.array(image)

        with st.spinner("🔎 Detecting... Please wait..."):
            results = model(frame, verbose=False)
            annotated = results[0].plot()

        # Count masks and no masks
        classes = results[0].boxes.cls.tolist()
        mask_count = sum(1 for c in classes if int(c) == 0)
        no_mask_count = sum(1 for c in classes if int(c) == 1)
        total = mask_count + no_mask_count

        if total > 0:
            st.success(f"✅ Masked Faces: {mask_count} | ❌ No Mask Faces: {no_mask_count}")
        else:
            st.warning("⚠️ No faces detected.")

        st.image(annotated, caption="🧠 Detection Result", use_container_width=True)
