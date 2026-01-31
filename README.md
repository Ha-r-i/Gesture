🖐️ Real-Time Gesture-to-Speech System

This project implements a real-time hand gesture recognition system that converts recognized hand gestures into spoken words using computer vision and machine learning.
The system leverages MediaPipe Hands for hand landmark detection and a RandomForestClassifier for gesture classification.

🔧 Prerequisites

Python 3.7+

Install required dependencies:

pip install -r requirements.txt


MediaPipe Model File
The system requires hand_landmarker.task for hand landmark detection.
Ensure this file exists in the project root directory.

If missing, download it from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

📁 Project Structure
.
├── collect_data.py        # Collects hand landmark data for gestures
├── train_model.py         # Trains the gesture classification model
├── app.py                 # Runs real-time recognition and speech output
├── data/                  # Stores collected gesture data
├── model.p                # Trained RandomForest model (generated)
├── hand_landmarker.task   # MediaPipe hand landmark model
└── requirements.txt

🧠 System Overview

The system follows this processing pipeline:

Webcam Feed
 → MediaPipe Hand Detection
 → 21 Hand Landmark Extraction
 → Feature Engineering
 → RandomForestClassifier
 → Gesture Prediction
 → Text-to-Speech Output

Technologies Used

MediaPipe Hands – real-time hand landmark detection

scikit-learn (RandomForestClassifier) – gesture classification

OpenCV – webcam capture and visualization

pyttsx3 – offline text-to-speech synthesis

✋ Hand Landmark Features

MediaPipe detects 21 anatomical hand landmarks

Each landmark provides normalized (x, y) coordinates

Feature vector per frame:

21 landmarks × 2 coordinates = 42 features


Landmark normalization ensures robustness to hand position and camera resolution

🚀 How to Use
Step 1: Collect Gesture Data

Run the data collection script:

python collect_data.py


Steps:

Enter the gesture name (e.g., Hello, Yes, No)

Enter the number of samples (100 recommended)

A webcam window will open

Position your hand clearly in the frame

Press S to record each sample

Repeat for all gestures you want to recognize

Collected data is automatically stored in the data/ directory.

Step 2: Train the Model

After collecting data for at least two gestures, train the classifier:

python train_model.py


This script:

Loads collected hand landmark data

Splits the dataset into training and testing sets

Trains a RandomForestClassifier

Evaluates model accuracy

Saves the trained model as model.p

Step 3: Run the Real-Time System

Start the gesture recognition system:

python app.py


What happens:

The webcam feed is processed in real time

Hand landmarks are detected using MediaPipe

Gestures are classified using the trained model

Recognized gestures are converted into spoken output

Press Q to exit the application

🔊 Text-to-Speech Conversion

Predicted gesture labels are passed to pyttsx3

Speech synthesis runs offline without internet access

A confirmation mechanism ensures gestures are spoken only when consistently detected

⚠️ Troubleshooting

Camera not opening
Ensure no other application is using the webcam.

Model not found
Run train_model.py before executing app.py.

Low accuracy

Collect more gesture samples

Capture gestures at different angles and distances

Ensure consistent lighting conditions

📌 Notes

This project uses classical machine learning, not deep learning.

MediaPipe is used exclusively for hand landmark extraction, not gesture classification.

Random Forest was chosen due to:

Structured landmark-based features

Efficient training on small-to-medium datasets

Fast real-time inference performance

✅ Summary

This project demonstrates a complete real-time gesture-to-speech pipeline, combining:

Hand landmark detection

Feature engineering

Supervised machine learning

Practical system integration with text-to-speech
