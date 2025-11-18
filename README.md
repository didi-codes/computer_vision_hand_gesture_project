# Project Title: Real-Time Hand Gesture Recognition System

## Project Overview
This project focuses on developing a real-time hand gesture recognition system using computer vision and deep learning techniques. The system will classify hand gestures from live webcam input and display the predicted gesture in real time.

## Problem Statement
Hand gesture recognition provides an intuitive and touch-free way to interact with technology. This project aims to create a lightweight model capable of accurately detecting multiple gesture classes with minimal latency, enabling applications such as contactless interfaces, games, and accessibility tools.

## Objectives
- Develop a custom gesture dataset (5 classes minimum).
- Train a small Convolutional Neural Network for gesture classification.
- Achieve about 85% training accuracy and about 80% validation accuracy.
- Implement real-time video capture using OpenCV.
- Perform real-time gesture predictions with on-screen annotations.
- Deliver a working demo that runs at 10 FPS.

## Dataset Requirements
Dataset Requirements
- Minimum 5 gesture categories (Thumbs Up, Peace Sign, Fist, Open Palm, OK Sign).
- At least 200–400 images total (40–80 per class).
- Images must include real variations: lighting, angles, hand orientation.
- Dataset split: 80% training, 20% validation.

## Technical Requirements
- Model: Custom CNN or MobileNetV2 (transfer learning allowed).
- Frameworks: TensorFlow/Keras, OpenCV.
- Input resolution: 64×64 or 96×96 RGB.
- Real-time loop must classify each frame and overlay class label on screen.

## Future Iterations
Revise gesture collection script to
-Press 1, 2, 3, 4, 5 on the keyboard
-Each number corresponds to a gesture
-No need to restart the script
-Images go into the correct gesture folder automatically