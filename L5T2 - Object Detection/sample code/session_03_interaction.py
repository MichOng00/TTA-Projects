"""
Session 03: Basic Hand Interaction
==================================
Detect when hand enters specific regions on screen.

Learning objectives:
- Accessing hand landmark coordinates
- Region-based interaction detection
- Visual feedback for interactions
- Basic event triggering

Requirements:
    pip install opencv-python "mediapipe>=0.10.33"

Controls:
    Q / ESC — Quit
"""

import cv2
import mediapipe as mp
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    print("Session 03: Basic Hand Interaction")
    print("===================================")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Focus on one hand for simplicity
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_name = "Session 03: Hand Interaction"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # Define interaction regions
    regions = [
        {"name": "TOP LEFT", "rect": (50, 50, 200, 150), "color": (255, 0, 0), "active": False},
        {"name": "TOP RIGHT", "rect": (width-250, 50, 200, 150), "color": (0, 255, 0), "active": False},
        {"name": "BOTTOM CENTER", "rect": (width//2-100, height-200, 200, 150), "color": (0, 0, 255), "active": False},
    ]

    print("Press Q or ESC to quit")
    print("Move your index finger into the colored regions")
    print("Regions will light up when your finger enters them")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Reset region states
        for region in regions:
            region["active"] = False

        # Check hand interactions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip position (landmark 8)
                index_x = int(hand_landmarks.landmark[8].x * width)
                index_y = int(hand_landmarks.landmark[8].y * height)

                # Check each region
                for region in regions:
                    rx, ry, rw, rh = region["rect"]
                    if rx <= index_x <= rx + rw and ry <= index_y <= ry + rh:
                        region["active"] = True
                        print(f"Entered region: {region['name']}")

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Highlight index finger
                cv2.circle(frame, (index_x, index_y), 8, (255, 255, 0), -1)

        # Draw regions
        for region in regions:
            rx, ry, rw, rh = region["rect"]
            color = region["color"] if region["active"] else (100, 100, 100)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 3)

            # Region label
            label_color = (255, 255, 255) if region["active"] else (150, 150, 150)
            cv2.putText(frame, region["name"], (rx + 10, ry + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # Status text
        active_regions = [r["name"] for r in regions if r["active"]]
        if active_regions:
            status = f"Active: {', '.join(active_regions)}"
        else:
            status = "Move index finger into regions"

        cv2.putText(frame, status, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Session 03 completed!")

if __name__ == "__main__":
    main()