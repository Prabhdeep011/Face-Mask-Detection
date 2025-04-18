import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pygame
from PIL import Image

# Load YOLO model
model_path = "/Users/prabhdeepsingh/Desktop/prabh/Python Files/runs/detect/train/weights/best.pt"
model = YOLO("best.pt")

# Initialize pygame alarm
alarm_path = "/Users/prabhdeepsingh/Desktop/prabh/alarm.mp3"
if not pygame.mixer.get_init():
    pygame.mixer.init()
pygame.mixer.music.load(alarm_path)

# Page configuration
st.set_page_config(page_title="😷 Mask Detection App", layout="centered", page_icon="😷")
st.markdown("<h1 style='text-align: center;'>😷 Face Mask Detection System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Initialize session state
if "detecting" not in st.session_state:
    st.session_state.detecting = False
if "alarm_enabled" not in st.session_state:
    st.session_state.alarm_enabled = True

# Sidebar settings
with st.sidebar:
    st.markdown("## 🛠️ Settings")
    mode = st.radio("🎯 Choose Mode", ["Live Detection", "Test Image (Upload / Capture)"])
    st.markdown("---")

    if mode == "Live Detection":
        st.session_state.alarm_enabled = st.checkbox("🔔 Enable Alarm", value=st.session_state.alarm_enabled)

        if not st.session_state.detecting:
            if st.button("▶️ Start Live Detection", use_container_width=True):
                st.session_state.detecting = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Detection", use_container_width=True):
                st.session_state.detecting = False
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                st.success("✅ Detection Stopped.")
                st.rerun()

    st.markdown("---")
    st.markdown("Made with ❤️ by **Prabh**")

# Live Detection Mode
if mode == "Live Detection":
    st.subheader("🟢 Live Webcam Feed")
    status = "🟢 Running" if st.session_state.detecting else "🔴 Stopped"
    st.markdown(f"**Status:** {status}")

    if st.session_state.detecting:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠️ Cannot access webcam.")
            st.session_state.detecting = False
            st.stop()

        frame_area = st.empty()

        while st.session_state.detecting:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            boxes = results[0].boxes

            alert = False
            if boxes is not None and len(boxes) > 0:
                classes = boxes.cls.cpu().tolist()
                confs = boxes.conf.cpu().tolist()

                for box, cls, conf in zip(boxes.xyxy, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    confidence = f"{conf * 100:.1f}%"
                    label = "Masked" if int(cls) == 0 else "No Mask"
                    color = (0, 255, 0) if int(cls) == 0 else (0, 0, 255)

                    if int(cls) == 1:
                        alert = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({confidence})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Alarm control
            if alert and st.session_state.alarm_enabled:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                pygame.mixer.music.stop()

            cv2.putText(frame, "😷 Please wear a mask!", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_area.image(rgb_frame, channels="RGB", use_container_width=True)

        cap.release()

# Test Image Mode
elif mode == "Test Image (Upload / Capture)":
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
