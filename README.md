# Real-Time Gesture-to-Speech System

This project implements a real-time hand gesture recognition system that converts recognized gestures into spoken words using Computer Vision and Machine Learning.

## Prerequisites

1.  **Python 3.7+** (Works with 3.14!)
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model File**: The system requires `hand_landmarker.task`. It has been downloaded to the root directory. If missing, download it from [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

## Project Structure

-   `collect_data.py`: Script to collect training data for your custom gestures.
-   `train_model.py`: Script to train the Machine Learning model on the collected data.
-   `app.py`: The main application that runs the real-time recognition and Text-to-Speech.
-   `data/`: Directory where gesture data is stored (created automatically).
-   `hand_landmarker.task`: MediaPipe model file for hand landmark detection.

## How to Use

### Step 1: Collect Data
Run the data collection script to record your gestures.
```bash
python collect_data.py
```
1.  Enter the name of the gesture (e.g., "Hello", "Yes", "No").
2.  Enter the number of samples (100 is recommended).
3.  A window will open. Position your hand.
4.  Press **'S'** to start capturing (the counter will increase).
5.  Repeat for as many gestures as you want.

### Step 2: Train the Model
Once you have collected data for at least two gestures, train the model.
```bash
python train_model.py
```
This will generate a `model.p` file.

### Step 3: Run the System
Start the real-time recognition system.
```bash
python app.py
```
-   The system will detect your hand.
-   When a gesture is recognized and held stable, the system will speak the gesture name.
-   Press **'Q'** to quit.

## Troubleshooting
-   **Camera not opening**: Ensure no other application is using the webcam.
-   **Model not found**: Make sure you ran `train_model.py` successfully.
-   **Accuracy issues**: Try collecting more data with different hand angles and distances.
