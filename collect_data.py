import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

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

def collect_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        # Get class name from user
        class_name = input("Enter the name of the gesture you want to collect (or 'q' to quit): ").strip()
        if class_name.lower() == 'q':
            break

        try:
            num_samples = int(input(f"How many samples for '{class_name}'? (Recommended: 100): "))
        except ValueError:
            print("Invalid number. Defaulting to 100.")
            num_samples = 100

        data = []
        
        print(f"\nCollecting data for '{class_name}'.")
        print("1. Position your hand in front of the camera.")
        print("2. Press 'S' to start capturing.")
        print("3. Press 'Q' to quit this gesture.")

        capturing = False
        counter = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calculate timestamp in milliseconds
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            # Draw landmarks
            if detection_result.hand_landmarks:
                annotated_image = draw_landmarks_on_image(frame, detection_result)
                # Replace frame with annotated image for display
                frame = annotated_image

                if capturing:
                    # Extract landmarks
                    for hand_landmarks in detection_result.hand_landmarks:
                        # We only take the first hand detected to simplify
                        data_aux = []
                        x_ = []
                        y_ = []

                        for landmark in hand_landmarks:
                            x = landmark.x
                            y = landmark.y
                            x_.append(x)
                            y_.append(y)

                        # Normalize data relative to min x and y
                        for landmark in hand_landmarks:
                            x = landmark.x
                            y = landmark.y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                        data.append(data_aux)
                        counter += 1
            
            # Display status
            cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {counter}/{num_samples}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if not capturing:
                cv2.putText(frame, "Press 'S' to Start", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Capturing...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(25)
            if key == ord('s') and not capturing:
                capturing = True
            elif key == ord('q'):
                break
            
            if counter >= num_samples:
                break
        
        # Save data
        if len(data) > 0:
            file_path = os.path.join(DATA_DIR, f'{class_name}.pickle')
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(data)} samples to {file_path}")
        else:
            print("No data collected.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
