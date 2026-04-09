"""
Session 04: Pinch Gesture Detection
===================================
Detect pinch gestures between thumb and index finger.

Learning objectives:
- Pinch gesture recognition
- Distance calculations between landmarks
- Gesture state management
- Visual feedback for gestures

Requirements:
    pip install opencv-python "mediapipe>=0.10.33"

Controls:
    Q / ESC — Quit
"""

import cv2
import mediapipe as mp
import math
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices
TIP_INDEX = 8   # Index finger tip
TIP_THUMB = 4   # Thumb tip

PINCH_THRESHOLD = 50  # Distance in pixels for pinch detection

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main():
    print("Session 04: Pinch Gesture Detection")
    print("====================================")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_name = "Session 04: Pinch Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    print("Press Q or ESC to quit")
    print("Bring thumb and index finger close together to create a pinch")
    print("The pinch distance will be displayed")

    pinch_count = 0
    last_pinch_state = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_pinch = False
        pinch_distance = float('inf')

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get thumb and index finger positions
                thumb_x = int(hand_landmarks.landmark[TIP_THUMB].x * width)
                thumb_y = int(hand_landmarks.landmark[TIP_THUMB].y * height)
                index_x = int(hand_landmarks.landmark[TIP_INDEX].x * width)
                index_y = int(hand_landmarks.landmark[TIP_INDEX].y * height)

                # Calculate distance
                pinch_distance = calculate_distance((thumb_x, thumb_y), (index_x, index_y))

                # Check for pinch
                if pinch_distance < PINCH_THRESHOLD:
                    current_pinch = True

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Draw pinch visualization
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 255, 0), 2)

                # Draw circles at fingertips
                thumb_color = (0, 255, 0) if current_pinch else (255, 0, 0)
                index_color = (0, 255, 0) if current_pinch else (0, 0, 255)

                cv2.circle(frame, (thumb_x, thumb_y), 8, thumb_color, -1)
                cv2.circle(frame, (index_x, index_y), 8, index_color, -1)

                # Draw pinch center
                center_x = (thumb_x + index_x) // 2
                center_y = (thumb_y + index_y) // 2
                cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

        # Count pinch events (rising edge)
        if current_pinch and not last_pinch_state:
            pinch_count += 1
            print(f"Pinch detected! Total pinches: {pinch_count}")

        last_pinch_state = current_pinch

        # Display information
        status_color = (0, 255, 0) if current_pinch else (255, 255, 255)
        status_text = "PINCHING!" if current_pinch else "Bring fingers together"

        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if pinch_distance != float('inf'):
            distance_text = f"Distance: {pinch_distance:.1f}px"
            cv2.putText(frame, distance_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Pinch Count: {pinch_count}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Session 04 completed! Total pinches: {pinch_count}")

if __name__ == "__main__":
    main()