# agri-bird-jetson-inference

# Agri-Bird: Automated Precision Gantry & Embedded Vision

## 🏆 Awards & Status
* **1st Place Winner:** *9th IET-GCC Robotics Challenge (GCC Final Round), Nov 2025.*
* **Status:** Patent Pending (2025–2026).

## 📌 Project Overview
Agri-Bird is an automated gantry system designed for precision agriculture. This repository contains the **embedded vision software** deployed on an **NVIDIA Jetson Nano**. It performs real-time disease detection on crops (specifically lettuce) and communicates with the gantry's motor controllers to take targeted action.

## 🧠 System Architecture
* **AI Model:** MobileNet V3 (Transfer Learning), optimized to **ONNX** format for edge inference.
* **Performance:** Achieved **98.29% accuracy** in detecting crop diseases with <50ms latency per frame.
* **Communication Bridge:** A custom Python script maps inference results to hardware commands sent via **UART/I2C** to the motor control unit.
* **Interface:** Includes a lightweight web dashboard (`index.html`) for monitoring system status in real-time.

## 🛠️ Tech Stack
* **Hardware:** NVIDIA Jetson Nano, Precision Gantry
* **AI/Vision:** PyTorch, Torchvision, OpenCV, ONNX Runtime
* **Connectivity:** Python `serial` (UART), Flask (Web Interface)

## 📂 File Structure
* `main.py`: The core inference loop. Captures video, runs the model, and sends UART commands.
* `test.py`: Validation script to benchmark model accuracy.
* `lettuce_detector.onnx`: The trained MobileNet V3 model (optimized).
* `index.html`: Web-based operator dashboard.

