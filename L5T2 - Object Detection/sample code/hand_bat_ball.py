"""
Hand Bat Ball — Use your hands as bats to hit a bouncing ball
===========================================================
Move your hands to create virtual bats and keep the ball in play!

Requirements:
    pip install opencv-python "mediapipe>=0.10.33" numpy

    The script auto-downloads the hand_landmarker.task model (~8 MB) on first
    run and caches it next to this script as hand_landmarker.task

Controls:
    Q / ESC — Quit
    R       — Reset ball
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import os
import urllib.request

# ──────────────────────────────────────────────
#  MediaPipe Tasks API  (mediapipe >= 0.10)
# ──────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

BaseOptions        = mp_python.BaseOptions
HandLandmarker     = mp_vision.HandLandmarker
HandLandmarkerOpts = mp_vision.HandLandmarkerOptions

# Landmark indices (21-point skeleton)
TIP_INDEX = 8   # index finger tip

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
WINDOW_NAME = "Hand Bat Ball"
TARGET_W    = 1280
TARGET_H    = 720
BALL_RADIUS = 20
BAT_WIDTH   = 120
BAT_HEIGHT  = 20
GRAVITY     = 300  # pixels per second squared
BOUNCE_DAMPING = 0.7  # energy loss on bounces
FRICTION    = 0.98   # air resistance

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# ──────────────────────────────────────────────
#  Model bootstrap
# ──────────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[Hand Bat Ball] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Hand Bat Ball] Model saved to", MODEL_PATH)

# ──────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def draw_ball(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, (255, 255, 255), 2, cv2.LINE_AA)

def draw_bat(img, center, width, height, color):
    x, y = center
    x1 = int(x - width // 2)
    y1 = int(y - height // 2)
    x2 = int(x + width // 2)
    y2 = int(y + height // 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

def draw_text_centered(img, text, center, font_scale=0.7,
                       color=(255, 255, 255), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = center[0] - tw // 2
    y = center[1] + th // 2
    cv2.putText(img, text, (x, y), font, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale,
                color, thickness, cv2.LINE_AA)

# ──────────────────────────────────────────────
#  Ball class
# ──────────────────────────────────────────────

class Ball:
    def __init__(self, pos):
        self.pos = list(pos)
        self.vel = [random.uniform(-150, 150), random.uniform(-50, 100)]  # More realistic starting velocities
        self.radius = BALL_RADIUS
        self.game_over = False

    def update(self, dt, w, h, bats):
        if self.game_over:
            return

        # Apply gravity
        self.vel[1] += GRAVITY * dt

        # Apply air resistance
        self.vel[0] *= FRICTION
        self.vel[1] *= FRICTION

        # Update position
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

        # Bounce off walls (except bottom)
        if self.pos[0] - self.radius < 0:
            self.pos[0] = self.radius
            self.vel[0] = abs(self.vel[0]) * BOUNCE_DAMPING
        if self.pos[0] + self.radius > w:
            self.pos[0] = w - self.radius
            self.vel[0] = -abs(self.vel[0]) * BOUNCE_DAMPING
        if self.pos[1] - self.radius < 0:
            self.pos[1] = self.radius
            self.vel[1] = abs(self.vel[1]) * BOUNCE_DAMPING

        # Check if ball hits bottom (game over)
        if self.pos[1] + self.radius > h:
            self.game_over = True
            self.pos[1] = h - self.radius
            self.vel = [0, 0]
            return

        # Check collision with bats (bats act as obstacles with bounce)
        for bat in bats:
            if self.collides_with_bat(bat):
                # Determine collision direction and bounce
                bat_center_x = bat['x']
                bat_center_y = bat['y']
                dx = self.pos[0] - bat_center_x
                dy = self.pos[1] - bat_center_y

                if abs(dx) > abs(dy):
                    # Horizontal collision - bounce horizontally
                    if dx > 0:
                        self.pos[0] = bat_center_x + bat['width'] // 2 + self.radius
                    else:
                        self.pos[0] = bat_center_x - bat['width'] // 2 - self.radius
                    self.vel[0] = -self.vel[0] * BOUNCE_DAMPING
                else:
                    # Vertical collision - bounce vertically
                    if dy > 0:
                        self.pos[1] = bat_center_y + bat['height'] // 2 + self.radius
                    else:
                        self.pos[1] = bat_center_y - bat['height'] // 2 - self.radius
                    self.vel[1] = -self.vel[1] * BOUNCE_DAMPING
                break

    def collides_with_bat(self, bat):
        ball_left = self.pos[0] - self.radius
        ball_right = self.pos[0] + self.radius
        ball_top = self.pos[1] - self.radius
        ball_bottom = self.pos[1] + self.radius

        bat_left = bat['x'] - bat['width'] // 2
        bat_right = bat['x'] + bat['width'] // 2
        bat_top = bat['y'] - bat['height'] // 2
        bat_bottom = bat['y'] + bat['height'] // 2

        return (ball_right > bat_left and ball_left < bat_right and
                ball_bottom > bat_top and ball_top < bat_bottom)

    def draw(self, img):
        cx, cy = int(self.pos[0]), int(self.pos[1])
        color = (0, 0, 255) if self.game_over else (0, 150, 255)
        draw_ball(img, (cx, cy), self.radius, color)

# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────

def run():
    ensure_model()

    # VIDEO mode: synchronous detect_for_video(), no callback needed.
    options = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ball = Ball((W // 2, H // 2))

    prev_time = time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, TARGET_W, TARGET_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        dt = min(now - prev_time, 0.05)
        prev_time = now
        timestamp_ms = int(now * 1000)

        # ── Hand detection via Tasks API ───────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Create bats from hand positions
        bats = []
        for hand_lms in result.hand_landmarks:
            index_x = int(hand_lms[TIP_INDEX].x * W)
            index_y = int(hand_lms[TIP_INDEX].y * H)

            # Use index finger tip as bat position
            bat_x = index_x
            bat_y = index_y

            bats.append({
                'x': bat_x,
                'y': bat_y,
                'width': BAT_WIDTH,
                'height': BAT_HEIGHT
            })

        # ── Update ────────────────────────────────────────────────
        ball.update(dt, W, H, bats)

        # ── Render ────────────────────────────────────────────────
        cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0, frame)

        ball.draw(frame)

        # Draw bats
        for bat in bats:
            draw_bat(frame, (bat['x'], bat['y']), bat['width'], bat['height'], (0, 255, 0))

        # HUD
        cv2.rectangle(frame, (0, 0), (W, 40), (10, 10, 20), -1)
        cv2.line(frame, (0, 40), (W, 40), (80, 60, 100), 1)
        title = "GAME OVER - R to restart" if ball.game_over else "HAND BAT BALL"
        draw_text_centered(frame, title,
                           (W // 2, 25), font_scale=0.75,
                           color=(200, 160, 255))
        draw_text_centered(frame,
                           "Use index fingers as bats to keep ball off bottom  |  R = reset  |  Q = quit",
                           (W // 2, H - 12), font_scale=0.42,
                           color=(140, 140, 160))

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            ball.pos = [W // 2, H // 2]
            ball.vel = [random.uniform(-150, 150), random.uniform(-50, 100)]
            ball.game_over = False

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()