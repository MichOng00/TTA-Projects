"""Minimal program to display MediaPipe hand landmarks"""
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Showing hand landmarks. Press ESC or 'q' to exit.")

try:
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        # Detect hands
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness_info.classification[0].label
                confidence = handedness_info.classification[0].score
                
                # Draw landmarks and connections
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(255, 0, 0), thickness=2
                    )
                )
                
                # Display hand label and confidence
                cv2.putText(
                    frame, f"{hand_label} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
        
        # Display frame
        cv2.imshow("Hand Landmarks", frame)
        
        # Exit on ESC or 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")
