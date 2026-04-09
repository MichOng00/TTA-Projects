"""
Session 05: Simple Object Pickup
================================
Create a simple object that can be picked up and moved with pinch gestures.

Learning objectives:
- Object representation and state management
- Pinch-to-grab interaction
- Object dragging with hand movement
- Basic collision detection

Requirements:
    pip install opencv-python "mediapipe>=0.10.33"

Controls:
    Q / ESC — Quit
    R       — Reset object position
"""

import cv2
import mediapipe as mp
import math
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

TIP_INDEX = 8
TIP_THUMB = 4
PINCH_THRESHOLD = 50

class SimpleObject:
    def __init__(self, x, y, radius=30):
        self.x = x
        self.y = y
        self.radius = radius
        self.grabbed = False
        self.color = (0, 150, 255)

    def draw(self, frame):
        color = (0, 255, 0) if self.grabbed else self.color
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)

    def contains_point(self, px, py):
        distance = math.sqrt((self.x - px)**2 + (self.y - py)**2)
        return distance <= self.radius

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("Session 05: Simple Object Pickup")
    print("=================================")

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

    window_name = "Session 05: Object Pickup"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # Create object in center
    obj = SimpleObject(width // 2, height // 2)

    print("Press Q or ESC to quit, R to reset object")
    print("Pinch near the blue circle to grab it")
    print("Move your hand while pinching to drag the object")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        pinching = False
        pinch_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get finger positions
                thumb_x = int(hand_landmarks.landmark[TIP_THUMB].x * width)
                thumb_y = int(hand_landmarks.landmark[TIP_THUMB].y * height)
                index_x = int(hand_landmarks.landmark[TIP_INDEX].x * width)
                index_y = int(hand_landmarks.landmark[TIP_INDEX].y * height)

                # Check for pinch
                distance = calculate_distance((thumb_x, thumb_y), (index_x, index_y))
                if distance < PINCH_THRESHOLD:
                    pinching = True
                    pinch_pos = ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Highlight pinch
                if pinching:
                    cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
                    cv2.circle(frame, pinch_pos, 5, (0, 255, 0), -1)

        # Handle object interaction
        if pinching and pinch_pos:
            if not obj.grabbed and obj.contains_point(pinch_pos[0], pinch_pos[1]):
                obj.grabbed = True
                print("Object grabbed!")
            elif obj.grabbed:
                # Drag object with pinch position
                obj.x = pinch_pos[0]
                obj.y = pinch_pos[1]
        else:
            if obj.grabbed:
                obj.grabbed = False
                print("Object released")

        # Draw object
        obj.draw(frame)

        # Status display
        status = "Grab the blue circle with pinch gesture"
        if obj.grabbed:
            status = "Dragging object - release pinch to drop"

        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if pinching:
            cv2.putText(frame, "PINCHING", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            obj.x = width // 2
            obj.y = height // 2
            obj.grabbed = False
            print("Object reset")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Session 05 completed!")

if __name__ == "__main__":
    main()