"""
Session 07: Multiple Objects and Collisions
===========================================
Create multiple physics objects that can collide with each other.

Learning objectives:
- Managing multiple objects
- Object-to-object collision detection
- Collision response between objects
- Object lifecycle management

Requirements:
    pip install opencv-python "mediapipe>=0.10.33"

Controls:
    Q / ESC — Quit
    R       — Reset all objects
    SPACE   — Launch all objects
    C       — Create new object at mouse position
"""

import cv2
import mediapipe as mp
import math
import time
import random
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

TIP_INDEX = 8
TIP_THUMB = 4
PINCH_THRESHOLD = 50

GRAVITY = 400
BOUNCE_DAMPING = 0.8
FRICTION = 0.99

class PhysicsObject:
    def __init__(self, x, y, radius=None, color=None):
        self.x = x
        self.y = y
        self.radius = radius or random.randint(20, 40)
        self.vx = random.uniform(-100, 100)
        self.vy = random.uniform(-50, 50)
        self.grabbed = False
        self.color = color or (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.id = id(self)  # unique identifier

    def update(self, dt, width, height, objects):
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

        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = abs(self.vy) * BOUNCE_DAMPING

        # Ground collision
        if self.y + self.radius > height:
            self.y = height - self.radius
            self.vy = -self.vy * BOUNCE_DAMPING
            self.vx *= 0.9

            if abs(self.vy) < 50:
                self.vy = 0
            if abs(self.vx) < 20:
                self.vx = 0

        # Check collisions with other objects
        for other in objects:
            if other.id != self.id and not other.grabbed:
                self.check_collision(other)

    def check_collision(self, other):
        # Calculate distance between centers
        dx = other.x - self.x
        dy = other.y - self.y
        distance = math.sqrt(dx*dx + dy*dy)

        min_distance = self.radius + other.radius

        if distance < min_distance and distance > 0:
            # Collision detected - separate objects
            overlap = min_distance - distance
            separation_x = (dx / distance) * overlap * 0.5
            separation_y = (dy / distance) * overlap * 0.5

            self.x -= separation_x
            self.y -= separation_y
            other.x += separation_x
            other.y += separation_y

            # Exchange velocities (simplified elastic collision)
            self.vx, other.vx = other.vx * BOUNCE_DAMPING, self.vx * BOUNCE_DAMPING
            self.vy, other.vy = other.vy * BOUNCE_DAMPING, self.vy * BOUNCE_DAMPING

    def draw(self, frame):
        color = (0, 255, 0) if self.grabbed else self.color
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)

    def contains_point(self, px, py):
        distance = math.sqrt((self.x - px)**2 + (self.y - py)**2)
        return distance <= self.radius

    def launch(self, force_x=0, force_y=-300):
        self.vx += force_x
        self.vy += force_y

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events to create objects"""
    if event == cv2.EVENT_LBUTTONDOWN:
        objects = param['objects']
        width, height = param['size']
        # Create new object at click position
        new_obj = PhysicsObject(x, y)
        objects.append(new_obj)
        print(f"Created new object at ({x}, {y})")

def main():
    print("Session 07: Multiple Objects and Collisions")
    print("============================================")

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

    window_name = "Session 07: Multiple Objects"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # Set up mouse callback
    callback_data = {'objects': [], 'size': (width, height)}
    cv2.setMouseCallback(window_name, mouse_callback, callback_data)

    # Create initial objects
    objects = [
        PhysicsObject(width // 4, height // 2, 30, (255, 100, 100)),
        PhysicsObject(3 * width // 4, height // 2, 25, (100, 255, 100)),
        PhysicsObject(width // 2, height // 4, 35, (100, 100, 255)),
    ]
    callback_data['objects'] = objects

    print("Controls:")
    print("  Q/ESC - Quit")
    print("  R - Reset all objects")
    print("  SPACE - Launch all objects")
    print("  Left click - Create new object")
    print("  Pinch - Grab and drag objects")

    prev_time = time.time()
    grabbed_object = None

    while True:
        current_time = time.time()
        dt = min(current_time - prev_time, 0.05)
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
            if grabbed_object is None:
                # Find object to grab
                for obj in objects:
                    if obj.contains_point(pinch_pos[0], pinch_pos[1]):
                        grabbed_object = obj
                        obj.grabbed = True
                        obj.vx = 0
                        obj.vy = 0
                        print("Object grabbed!")
                        break
            elif grabbed_object:
                # Move grabbed object
                grabbed_object.x = pinch_pos[0]
                grabbed_object.y = pinch_pos[1]
        else:
            if grabbed_object:
                grabbed_object.grabbed = False
                grabbed_object = None
                print("Object released")

        # Update all objects
        for obj in objects:
            obj.update(dt, width, height, objects)

        # Draw ground
        cv2.rectangle(frame, (0, height - 10), (width, height), (100, 100, 100), -1)

        # Draw all objects
        for obj in objects:
            obj.draw(frame)

        # Status display
        cv2.putText(frame, f"Objects: {len(objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if pinching:
            cv2.putText(frame, "PINCHING", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "Click to create objects", (10, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            objects.clear()
            objects.extend([
                PhysicsObject(width // 4, height // 2, 30, (255, 100, 100)),
                PhysicsObject(3 * width // 4, height // 2, 25, (100, 255, 100)),
                PhysicsObject(width // 2, height // 4, 35, (100, 100, 255)),
            ])
            grabbed_object = None
            print("Objects reset")
        elif key == 32:  # Space
            for obj in objects:
                if not obj.grabbed:
                    obj.launch(random.uniform(-100, 100), random.uniform(-200, -100))
            print("All objects launched!")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Session 07 completed! Created {len(objects)} objects total")

if __name__ == "__main__":
    main()