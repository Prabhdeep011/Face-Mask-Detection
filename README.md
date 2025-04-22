# Face Mask Detection App

## Project Overview

This project is a **Face Mask Detection App** built using **YOLOv8**, **OpenCV**, and **Streamlit**. The app utilizes webcam footage to detect whether individuals are wearing a face mask in real-time. If a person is detected without a mask, an alarm sound is triggered as an alert. The YOLOv8 model is used for its accuracy and speed in detecting faces and classifying them based on mask status.

## Features

- **Real-time Face Mask Detection**: Detects whether a person is wearing a face mask using live webcam footage.
- **Alarm Sound**: Plays an alarm when a person is detected without a face mask.
- **Streamlit Interface**: A user-friendly, interactive interface that allows easy access to webcam functionality and results.

## Problem Faced During Deployment

While the app works well locally, deploying it on cloud platforms such as **Streamlit Cloud**, **Heroku**, and **Replit** comes with several challenges:

1. **Webcam Access Issues**: **Streamlit** and other cloud platforms restrict access to webcam functionality, which makes live detection with webcam inputs unfeasible on cloud-based environments.
   
2. **Alarm Sound Not Playing**: The alarm sound functionality works correctly on the local machine but does not play as expected when the app is deployed to the cloud. This is due to limitations on audio processing and playback in cloud services.









here's the link:
https://face-mask-detection-ew4o89jpyffupwkmnjsjp7.streamlit.app/
