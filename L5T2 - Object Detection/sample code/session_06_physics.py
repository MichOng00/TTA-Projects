"""
Session 06: Physics and Gravity
===============================
Add realistic physics with gravity and bouncing to the pickup object.

Learning objectives:
- Basic physics simulation (gravity, velocity)
- Collision response with boundaries
- Time-based updates
- Energy loss on bounces

Requirements:
    pip install opencv-python "mediapipe>=0.10.33"

Controls:
    Q / ESC — Quit
    R       — Reset object
    SPACE   — Launch object upward
"""

import cv2
import mediapipe as mp
import math
import time
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

TIP_INDEX = 8
TIP_THUMB = 4
PINCH_THRESHOLD = 50

GRAVITY = 400  # pixels per second squared
BOUNCE_DAMPING = 0.8  # energy loss on bounce
FRICTION = 0.99  # air resistance

class PhysicsObject:
    def __init__(self, x, y, radius=30):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 0  # velocity x
        self.vy = 0  # velocity y
        self.grabbed = False
        self.color = (0, 150, 255)

    def update(self, dt, width, height):
        if self.grabbed:
            return

        # Apply gravity
        self.vy += GRAVITY * dt

        # Apply air resistance
        self.vx *= FRICTION
        self.vy *= FRICTION

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off walls
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx) * BOUNCE_DAMPING
        elif self.x + self.radius > width:
            self.x = width - self.radius
            self.vx = -abs(self.vx) * BOUNCE_DAMPING

        # Bounce off top
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = abs(self.vy) * BOUNCE_DAMPING

        # Stop at bottom (ground)
        if self.y + self.radius > height:
            self.y = height - self.radius
            self.vy = -self.vy * BOUNCE_DAMPING
            self.vx *= 0.9  # extra friction on ground

            # Stop small movements
            if abs(self.vy) < 50:
                self.vy = 0
            if abs(self.vx) < 20:
                self.vx = 0

    def draw(self, frame):
        color = (0, 255, 0) if self.grabbed else self.color
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)

        # Draw velocity vector when moving
        if not self.grabbed and (abs(self.vx) > 10 or abs(self.vy) > 10):
            end_x = int(self.x + self.vx * 0.1)
            end_y = int(self.y + self.vy * 0.1)
            cv2.arrowedLine(frame, (int(self.x), int(self.y)), (end_x, end_y),
                          (255, 255, 0), 2, tipLength=0.3)

    def contains_point(self, px, py):
        distance = math.sqrt((self.x - px)**2 + (self.y - py)**2)
        return distance <= self.radius

    def launch(self, force_x=0, force_y=-300):
        """Apply an impulse force"""
        self.vx += force_x
        self.vy += force_y

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("Session 06: Physics and Gravity")
    print("===============================")

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

    window_name = "Session 06: Physics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # Create physics object
    obj = PhysicsObject(width // 2, height // 2 - 100)

    print("Press Q or ESC to quit, R to reset, SPACE to launch")
    print("Pinch to grab and drag the ball")
    print("Watch it fall with gravity and bounce!")

    prev_time = time.time()

    while True:
        current_time = time.time()
        dt = min(current_time - prev_time, 0.05)  # Cap delta time
        prev_time = current_time

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
                thumb_x = int(hand_landmarks.landmark[TIP_THUMB].x * width)
                thumb_y = int(hand_landmarks.landmark[TIP_THUMB].y * height)
                index_x = int(hand_landmarks.landmark[TIP_INDEX].x * width)
                index_y = int(hand_landmarks.landmark[TIP_INDEX].y * height)

                distance = calculate_distance((thumb_x, thumb_y), (index_x, index_y))
                if distance < PINCH_THRESHOLD:
                    pinching = True
                    pinch_pos = ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                if pinching:
                    cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
                    cv2.circle(frame, pinch_pos, 5, (0, 255, 0), -1)

        # Handle object interaction
        if pinching and pinch_pos:
            if not obj.grabbed and obj.contains_point(pinch_pos[0], pinch_pos[1]):
                obj.grabbed = True
                obj.vx = 0  # Stop velocity when grabbed
                obj.vy = 0
                print("Object grabbed!")
            elif obj.grabbed:
                # Move object with hand
                obj.x = pinch_pos[0]
                obj.y = pinch_pos[1]
        else:
            if obj.grabbed:
                obj.grabbed = False
                print("Object released")

        # Update physics
        obj.update(dt, width, height)

        # Draw ground
        cv2.rectangle(frame, (0, height - 10), (width, height), (100, 100, 100), -1)

        # Draw object
        obj.draw(frame)

        # Status display
        status = "Pinch to grab the ball"
        if obj.grabbed:
            status = "Dragging - release to drop"

        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if pinching:
            cv2.putText(frame, "PINCHING", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Physics info
        cv2.putText(frame, f"Vel: ({obj.vx:.0f}, {obj.vy:.0f})", (10, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            obj.x = width // 2
            obj.y = height // 2 - 100
            obj.vx = 0
            obj.vy = 0
            obj.grabbed = False
            print("Object reset")
        elif key == 32:  # Space
            if not obj.grabbed:
                obj.launch()
                print("Object launched!")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Session 06 completed!")

if __name__ == "__main__":
    main()