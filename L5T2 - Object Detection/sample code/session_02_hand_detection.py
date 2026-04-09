"""
Session 02: Hand Detection
==========================
Add MediaPipe hand detection to display hand landmarks.

Learning objectives:
- MediaPipe setup and initialization
- Hand landmark detection
- Drawing landmarks on the frame
- Understanding hand tracking data

Requirements:
    pip install opencv-python "mediapipe>=0.10.33"

Controls:
    Q / ESC — Quit
"""

import cv2
import mediapipe as mp
import sys
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    print("Session 02: Hand Detection")
    print("===========================")

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Create video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create window
    window_name = "Session 02: Hand Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    print("Press Q or ESC to quit")
    print("Move your hand in front of the camera")
    print("You should see hand landmarks drawn on your hand")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Count detected hands
                num_hands = len(results.multi_hand_landmarks)
                cv2.putText(frame, f"Hands detected: {num_hands}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display frame
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Session 02 completed!")

if __name__ == "__main__":
    main()