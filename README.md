# 🔴 Real-Time Violence Detection Using CNN and CBAM

An intelligent computer vision system that detects **violent and non-violent activities** from surveillance videos and live webcam streams. The system compares traditional machine learning, CNN, and attention-based CNN approaches and provides **real-time visual and audio alerts**.

## 📌 Models Used

### 1. LBP + Linear SVM

LBP extracts **handcrafted local texture features** from video frames, which are classified using a Linear SVM.
It serves as a traditional machine learning baseline and achieves **75.12% accuracy**.

### 2. CNN

The CNN automatically learns **hierarchical visual features** such as edges, textures, body postures, and fighting patterns from images.
It improves the baseline performance and achieves **83% accuracy**.

### 3. CNN + CBAM ⭐ Proposed Model

CBAM adds **Channel Attention and Spatial Attention**, helping the CNN focus on important features and regions related to violent activity.
The proposed model achieves the best performance with **95% accuracy**.

## 📊 Dataset

* **1,000** Violence videos
* **1,000** Non-Violence videos
* **23,952** extracted frames
* Image size: **224 × 224 × 3**
* Training: **80%**
* Testing: **20%**

## 📈 Results

| Model            | Accuracy |
| ---------------- | -------: |
| LBP + Linear SVM |   75.12% |
| CNN              |      83% |
| **CNN + CBAM**   |  **95%** |

## 🛠️ Technologies

**Python • TensorFlow • Keras • OpenCV • Scikit-learn • Flask**

## 🚨 Features

* Violence / Non-Violence classification
* Video upload detection
* Real-time webcam monitoring
* Visual warning alerts
* Audio alarm

## 🔄 Workflow


Video / Webcam --> Frame Extraction --> Preprocessing --> CNN + CBAM --> Violence Detection --> Visual + Audio Alert

## 🔮 Future Scope

* Live CCTV integration
* Larger and more diverse datasets
* CNN-LSTM / 3D CNN / Vision Transformers
* Edge-device deployment
* Automatic emergency notifications
