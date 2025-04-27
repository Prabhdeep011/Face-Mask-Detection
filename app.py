import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model_path = "best.pt"  # Make sure 'best.pt' is in the same folder or give full path
model = YOLO(model_path)

# Streamlit page setup
st.set_page_config(page_title="😷 Mask Detection App", layout="centered", page_icon="😷")
st.markdown("<h1 style='text-align: center;'>😷 Face Mask Detection System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.markdown("## 🛠️ Settings")
    mode = st.radio("🎯 Choose Mode", ["Test Image (Upload / Capture)"])
    st.markdown("---")

    # About section toggle
    show_about = st.checkbox("📖 About", value=False)

# Display About details when checkbox is checked
if show_about:
    st.markdown("## 📖Details")
    st.markdown(
        """
        **Face Mask Detection System** is a machine learning model built to detect whether individuals are wearing face masks or not using computer vision techniques.

        **Technology Stack:**
        - **Model:** YOLOv8 (Ultralytics)
        - **Framework:** Streamlit
        - **Libraries:** OpenCV, NumPy, PIL

        **How it works:**
        - This app uses a pre-trained YOLOv8 model to detect faces in images. 
        - The model classifies each face as either **Masked** or **No Mask**.
        - It provides an interactive UI for testing with image uploads and visualizes detection results.

        **Purpose:**
        - This tool helps in monitoring mask usage for safety purposes, especially in public settings.
        """
    )
    st.markdown("---")

# Test Image Mode
if mode == "Test Image (Upload / Capture)":
    st.subheader("📸 Test with Image")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run detection
        if st.button("🚀 Run Detection"):
            with st.spinner("🔎 Detecting... Please wait..."):
                results = model(frame, verbose=False)
                annotated = results[0].plot()

            # Count Mask / No Mask
            classes = results[0].boxes.cls.tolist()
            mask_count = sum(1 for c in classes if int(c) == 0)
            no_mask_count = sum(1 for c in classes if int(c) == 1)

            if (mask_count + no_mask_count) > 0:
                st.success(f"✅ Masked Faces: {mask_count} | ❌ No Mask Faces: {no_mask_count}")
            else:
                st.warning("⚠️ No faces detected.")

            # Show the final image
            st.image(annotated, caption="🧠 Detection Result", use_container_width=True)
        else:
            st.info("Click the '🚀 Run Detection' button after uploading an image.")

    else:
        st.info("Please upload an image to run detection.")
