import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

# Load model
model_path = 'model.p'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
else:
    model = None
    print("Warning: model.p not found. Prediction will not work until you collect data and train.")

# Text-to-Speech Setup
def speak_text(text):
    """Function to be run in a separate thread to avoid blocking the video loop"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

last_spoken_time = 0
last_prediction = None
confirmation_frames = 0
CONFIRMATION_THRESHOLD = 10  # Number of frames to confirm a gesture

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Draw the landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
            
        # Draw connections (simplified)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (0, 17) # Wrist to pinky
        ]
        
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            x1 = int(start_point.x * annotated_image.shape[1])
            y1 = int(start_point.y * annotated_image.shape[0])
            x2 = int(end_point.x * annotated_image.shape[1])
            y2 = int(end_point.y * annotated_image.shape[0])
            
            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
    return annotated_image

def main():
    global last_prediction, confirmation_frames, last_spoken_time
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Starting Gesture Recognition System...")
    print("Press 'q' to quit.")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        timestamp_ms = int((time.time() - start_time) * 1000)

        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        current_prediction = None
        
        if detection_result.hand_landmarks:
            frame = draw_landmarks_on_image(frame, detection_result)
            
            for hand_landmarks in detection_result.hand_landmarks:
                if model is not None:
                    # Extract features
                    data_aux = []
                    x_ = []
                    y_ = []

                    for landmark in hand_landmarks:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)

                    for landmark in hand_landmarks:
                        x = landmark.x
                        y = landmark.y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Predict
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        current_prediction = prediction[0]
                    except Exception as e:
                        print(f"Prediction Error: {e}")
                
                # We only process the first hand for now
                break
        
        # Logic for stability and TTS
        if current_prediction:
            cv2.putText(frame, f"Gesture: {current_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if current_prediction == last_prediction:
                confirmation_frames += 1
            else:
                confirmation_frames = 0
                last_prediction = current_prediction
            
            if confirmation_frames == CONFIRMATION_THRESHOLD:
                # Speak only if enough time has passed since last speech (e.g., 2 seconds)
                if time.time() - last_spoken_time > 2.0:
                    print(f"Speaking: {current_prediction}")
                    threading.Thread(target=speak_text, args=(current_prediction,), daemon=True).start()
                    last_spoken_time = time.time()
        else:
            confirmation_frames = 0
            last_prediction = None

        cv2.imshow('Gesture to Speech', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
