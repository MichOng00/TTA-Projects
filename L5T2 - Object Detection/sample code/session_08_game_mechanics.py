"""
Session 08: Game Mechanics - Scoring and Levels
===============================================
Add scoring system, levels, and game objectives to make it a complete game.

Learning objectives:
- Game state management
- Scoring systems
- Level progression
- Win/lose conditions
- UI for game stats

Requirements:
    pip install opencv-python "mediapipe>=0.10.33"

Controls:
    Q / ESC — Quit
    R       — Restart level
    N       — Next level (when completed)
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

class Target:
    def __init__(self, x, y, radius=25, points=10):
        self.x = x
        self.y = y
        self.radius = radius
        self.points = points
        self.hit = False
        self.color = (255, 255, 100)

    def draw(self, frame):
        if not self.hit:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, 3)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius // 2, self.color, -1)
            # Points display
            cv2.putText(frame, str(self.points), (int(self.x - 10), int(self.y + 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def contains_point(self, px, py):
        distance = math.sqrt((self.x - px)**2 + (self.y - py)**2)
        return distance <= self.radius

class PhysicsObject:
    def __init__(self, x, y, radius=30, color=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 0
        self.vy = 0
        self.grabbed = False
        self.color = color or (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.id = id(self)

    def update(self, dt, width, height, objects, targets):
        if self.grabbed:
            return

        # Apply gravity
        self.vy += GRAVITY * dt
        self.vx *= FRICTION
        self.vy *= FRICTION

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Wall collisions
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

        # Check target collisions
        for target in targets:
            if not target.hit and target.contains_point(self.x, self.y):
                target.hit = True
                return target.points  # Return points scored

        # Object collisions
        for other in objects:
            if other.id != self.id and not other.grabbed:
                self.check_collision(other)

        return 0

    def check_collision(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        min_distance = self.radius + other.radius

        if distance < min_distance and distance > 0:
            overlap = min_distance - distance
            separation_x = (dx / distance) * overlap * 0.5
            separation_y = (dy / distance) * overlap * 0.5

            self.x -= separation_x
            self.y -= separation_y
            other.x += separation_x
            other.y += separation_y

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

class GameLevel:
    def __init__(self, level_num):
        self.level_num = level_num
        self.targets = []
        self.objects = []
        self.score_required = 0
        self.time_limit = 60  # seconds
        self.start_time = time.time()

        self.setup_level()

    def setup_level(self):
        # Create targets and objects based on level
        num_targets = min(3 + self.level_num, 8)
        num_objects = min(2 + self.level_num // 2, 5)

        # Target positions (top area)
        target_positions = [
            (200, 100), (400, 120), (600, 100), (300, 80), (500, 90),
            (150, 150), (650, 150), (400, 60)
        ]

        for i in range(num_targets):
            if i < len(target_positions):
                x, y = target_positions[i]
                points = 10 + (self.level_num - 1) * 5
                self.targets.append(Target(x, y, 25, points))
                self.score_required += points

        # Object positions (bottom area)
        object_positions = [
            (200, 400), (400, 420), (600, 400), (300, 380), (500, 390)
        ]

        for i in range(num_objects):
            if i < len(object_positions):
                x, y = object_positions[i]
                self.objects.append(PhysicsObject(x, y))

    def is_completed(self, current_score):
        return current_score >= self.score_required

    def time_remaining(self):
        elapsed = time.time() - self.start_time
        return max(0, self.time_limit - elapsed)

    def is_time_up(self):
        return self.time_remaining() <= 0

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("Session 08: Game Mechanics")
    print("==========================")

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

    window_name = "Session 08: Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # Game state
    current_level = 1
    score = 0
    level = GameLevel(current_level)
    game_state = "playing"  # playing, level_complete, game_over

    print("Controls:")
    print("  Q/ESC - Quit")
    print("  R - Restart level")
    print("  N - Next level (when completed)")
    print("Goal: Hit all targets with objects to score points!")

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
        if pinching and pinch_pos and game_state == "playing":
            if grabbed_object is None:
                for obj in level.objects:
                    if obj.contains_point(pinch_pos[0], pinch_pos[1]):
                        grabbed_object = obj
                        obj.grabbed = True
                        obj.vx = 0
                        obj.vy = 0
                        break
            elif grabbed_object:
                grabbed_object.x = pinch_pos[0]
                grabbed_object.y = pinch_pos[1]
        else:
            if grabbed_object:
                grabbed_object.grabbed = False
                grabbed_object = None

        # Update game objects
        if game_state == "playing":
            points_scored = 0
            for obj in level.objects:
                points_scored += obj.update(dt, width, height, level.objects, level.targets)

            score += points_scored
            if points_scored > 0:
                print(f"Scored {points_scored} points! Total: {score}")

            # Check win/lose conditions
            if level.is_completed(score):
                game_state = "level_complete"
                print(f"Level {current_level} completed!")
            elif level.is_time_up():
                game_state = "game_over"
                print("Time's up!")

        # Draw everything
        # Ground
        cv2.rectangle(frame, (0, height - 10), (width, height), (100, 100, 100), -1)

        # Targets
        for target in level.targets:
            target.draw(frame)

        # Objects
        for obj in level.objects:
            obj.draw(frame)

        # UI
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Level: {current_level}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score}/{level.score_required}", (200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {level.time_remaining():.1f}s", (400, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if game_state == "level_complete":
            cv2.putText(frame, "LEVEL COMPLETE! Press N for next level", (width//2 - 200, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        elif game_state == "game_over":
            cv2.putText(frame, "TIME'S UP! Press R to restart", (width//2 - 150, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if pinching:
            cv2.putText(frame, "PINCHING", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            level = GameLevel(current_level)
            score = 0
            game_state = "playing"
            grabbed_object = None
            print(f"Level {current_level} restarted")
        elif key == ord('n') and game_state == "level_complete":
            current_level += 1
            level = GameLevel(current_level)
            score = 0
            game_state = "playing"
            grabbed_object = None
            print(f"Starting level {current_level}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Session 08 completed! Reached level {current_level} with score {score}")

if __name__ == "__main__":
    main()