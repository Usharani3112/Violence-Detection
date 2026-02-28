# 🎯 Real-Time Violence Detection System using CNN + CBAM

An AI-powered web-based system for detecting violent activities in
real-time using Deep Learning, Computer Vision, and Flask.

------------------------------------------------------------------------

# 📌 1. Project Overview & 📂 Project Structure

This project implements a deep learning-based **Violence Detection
System** capable of:

-   📹 Detecting violence from live webcam feed
-   📁 Analyzing uploaded videos
-   🔔 Triggering real-time alarm alerts
-   📊 Providing confidence scores and frame-level statistics

The model is trained using the **Real Life Violence Dataset** and
enhanced with **CBAM (Convolutional Block Attention Module)** to improve
spatial and channel attention.

Violence-Detection/
│
├── app.py # Flask Application
├── index.html # Frontend UI
├── config.json # Detection thresholds
├── model/
│ ├── complete_model.keras # Trained model
│ ├── model_info.json
│ └── *.weights.h5
│
├── final_complete_training.ipynb
├── violence_detection_system.ipynb
│
└── README.md

------------------------------------------------------------------------

# 📂 2. Dataset Used

## 📊 Real Life Violence Dataset

-   Two classes:
    -   Violence
    -   Non-Violence
-   Extracted frames used for CNN training
-   Balanced dataset for classification

Preprocessing steps: - Frame extraction from videos - Resizing to 224 ×
224 - Normalizing pixel values (0--1) - One-hot encoding labels

------------------------------------------------------------------------

# 🧠 3. Model Architecture

### 🔹 CNN Blocks

-   Conv2D
-   Batch Normalization
-   ReLU Activation
-   MaxPooling

### 🔹 CBAM Attention Module

-   Channel Attention
-   Spatial Attention
-   Enhances important feature focus

### 🔹 Final Layers

-   Global Average Pooling
-   Dense (256 units)
-   Dropout (0.5)
-   Softmax (2 classes)

Input Shape: 224 x 224 x 3\
Output: \[Non-Violence, Violence\]

------------------------------------------------------------------------

# 📓 4. Training Process (Notebook Files)

## 📘 violence_detection_system.ipynb

-   Dataset loading
-   Frame extraction
-   Data preprocessing
-   Model building
-   Initial training

## 📘 final_complete_training.ipynb

-   Hyperparameter tuning
-   Model checkpointing
-   Saving best weights
-   Accuracy evaluation
-   Exporting final model (.keras / .h5)

Saved Model Structure:

model/ complete_model.keras model_info.json \*.weights.h5

------------------------------------------------------------------------

# 🌐 5. Web Application Structure

## Backend: app.py

-   Flask server
-   Loads trained model
-   Webcam streaming
-   Video upload handling
-   Prediction API endpoints
-   Alarm trigger logic

## Frontend: index.html

-   Responsive UI
-   Live Webcam Tab
-   Upload Video Tab
-   Real-time alert animation
-   Audio alarm system

------------------------------------------------------------------------

# 🔍 6. Detection Logic

## Webcam Detection

-   Detect every N frames
-   Apply smoothing window
-   Trigger alarm if sustained violence probability \> threshold

## Video Detection

-   Sample fixed number of frames
-   Predict violence probability per frame
-   Calculate:
    -   Average probability
    -   Maximum probability
    -   Violence frame ratio
-   Final decision based on threshold and frame ratio

------------------------------------------------------------------------

# ⚙️ 7. Installation Guide

## Step 1: Clone Repository

git clone https://github.com/yourusername/VIOLENCE-DETECTION.git cd
VIOLENCE-DETECTION

## Step 2: Create Virtual Environment (Recommended)

python -m venv venv venv`\Scripts`{=tex}`\activate   `{=tex}(Windows)

## Step 3: Install Dependencies

pip install -r requirements.txt

If requirements.txt is not present: pip install flask tensorflow
opencv-python numpy werkzeug

## Step 4: Place Model Files

Ensure model files are inside the model/ folder.

## Step 5: Run Application

python app.py

Open in browser: http://127.0.0.1:5000

------------------------------------------------------------------------

# 🔌 8. API Endpoints

  Endpoint          Method   Description
  ----------------- -------- --------------------------
  /                 GET      Main UI
  /start_webcam     POST     Start webcam detection
  /stop_webcam      POST     Stop webcam detection
  /predict_video    POST     Upload and analyze video
  /check_violence   GET      Check live detection
  /health           GET      Server health check

------------------------------------------------------------------------

# 📊 9. Key Highlights

-   Attention-enhanced deep learning model
-   Real-time inference
-   Frame-level analysis
-   Alarm triggering mechanism
-   Configurable thresholds
-   Production-ready Flask backend

------------------------------------------------------------------------

# 🔮 10. Future Improvements

-   Cloud deployment (AWS / GCP)
-   Mobile application integration
-   Multi-camera support
-   SMS/Email alert system
-   Model optimization (TensorRT / ONNX)

------------------------------------------------------------------------

# 👨‍💻 Author

Developed as a Deep Learning & Computer Vision Project.\
Role: AI/ML Developer

------------------------------------------------------------------------

# ⚠️ Disclaimer

This system is built for academic and research purposes.\
Real-world deployment must comply with privacy and surveillance
regulations.

------------------------------------------------------------------------

⭐ If you found this project useful, consider giving it a star!
