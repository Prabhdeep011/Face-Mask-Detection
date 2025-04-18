import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import os
from PIL import Image

# Title and mode selection
st.title("😷 Face Mask Detection System")
mode = st.radio("Choose Detection Mode", ["Live Detection", "Test Image (Upload/Capture)"], horizontal=True)

# Check if running on Streamlit Cloud
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS", "1") == "1"

# Load YOLOv8 model
model_path = "best.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# Alarm setup
def play_alarm():
    if not IS_CLOUD:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load("alarm.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            st.warning(f"Alarm Error: {e}")
    else:
        st.warning("🚨 Person without mask detected!")
        with st.expander("🔊 Play alarm manually"):
            audio_file = open("alarm.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')

def stop_alarm():
    if not IS_CLOUD:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

# -------------------------- Live Detection --------------------------
if mode == "Live Detection":
    st.subheader("🔴 Live Webcam Feed with Mask Detection")
    start = st.button("Start Live Detection")
    stop = st.button("Stop Detection")
    alarm_toggle = st.checkbox("Enable Alarm", value=True)

    if start and not stop:
        if IS_CLOUD:
            st.error("Live webcam access is not supported on Streamlit Cloud. Please run locally.")
            st.stop()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam could not be accessed.")
            st.stop()

        frame_holder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            boxes = results[0].boxes
            classes = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            alert = False
            mask_count, no_mask_count = 0, 0

            for box, cls, conf in zip(boxes.xyxy, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                confidence = f"{conf * 100:.1f}%"
                label = "Masked" if int(cls) == 0 else "No Mask"
                color = (0, 255, 0) if int(cls) == 0 else (0, 0, 255)

                if int(cls) == 1:
                    alert = True
                    no_mask_count += 1
                else:
                    mask_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({confidence})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if alert and alarm_toggle:
                play_alarm()
            else:
                stop_alarm()

            cv2.putText(frame, "Please wear a mask and stay safe!", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_holder.image(rgb_frame, channels="RGB", use_container_width=True)

            if stop:
                break

        cap.release()
        stop_alarm()
        st.success("Live detection stopped.")

# ---------------------- Image Test / Upload -----------------------
elif mode == "Test Image (Upload/Capture)":
    st.subheader("📸 Upload or Capture an Image")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    capture = st.button("📷 Open Webcam to Capture")
    captured_image = None

    if capture:
        if IS_CLOUD:
            st.error("Webcam capture is not supported on Streamlit Cloud.")
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                st.image(frame, caption="Preview - Press Capture", channels="BGR")
                if st.button("✅ Capture This Image"):
                    captured_image = frame
                    st.success("Image captured successfully.")
            else:
                st.error("Failed to capture from webcam.")

    image_input = None
    if uploaded_image:
        image_input = Image.open(uploaded_image)
        image_input = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif captured_image is not None:
        image_input = captured_image

    if image_input is not None:
        results = model(image_input, verbose=False)
        boxes = results[0].boxes
        classes = boxes.cls.cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        mask_count, no_mask_count = 0, 0

        for box, cls, conf in zip(boxes.xyxy, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label = "Masked" if int(cls) == 0 else "No Mask"
            color = (0, 255, 0) if int(cls) == 0 else (0, 0, 255)

            if int(cls) == 0:
                mask_count += 1
            else:
                no_mask_count += 1

            cv2.rectangle(image_input, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_input, f"{label} ({conf * 100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        total = mask_count + no_mask_count
        accuracy = (mask_count / total) * 100 if total > 0 else 0

        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=f"Detection Completed. Accuracy: {accuracy:.2f}%", channels="RGB")

# -------------------------- Refresh --------------------------
if st.button("🔄 Refresh All"):
    st.rerun()
