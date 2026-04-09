"""
Simple Pickup — Basic hand-tracking pickup game
===============================================
Use your hand to PINCH and DRAG a ball around the screen.

Requirements:
    pip install opencv-python "mediapipe>=0.10.33" numpy

    The script auto-downloads the hand_landmarker.task model (~8 MB) on first
    run and caches it next to this script as hand_landmarker.task

Controls:
    Q / ESC — Quit
    R       — Reset ball position
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
TIP_THUMB = 4   # thumb tip

# Bone connections for hand skeleton overlay
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
WINDOW_NAME = "Simple Pickup"
TARGET_W    = 1280
TARGET_H    = 720
BALL_RADIUS = 50
PINCH_DIST_PX = 60

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
        print("[Simple Pickup] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Simple Pickup] Model saved to", MODEL_PATH)

# ──────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def lerp(a, b, t):
    return a + (b - a) * t

def draw_ball(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, (255, 255, 255), 2, cv2.LINE_AA)

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
        self.vel = [random.uniform(-100, 100), random.uniform(-100, 100)]
        self.grabbed = False

    def update(self, dt, w, h):
        if self.grabbed:
            return
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        pad = BALL_RADIUS + 10
        if self.pos[0] < pad:
            self.pos[0] = pad; self.vel[0] = abs(self.vel[0])
        if self.pos[0] > w - pad:
            self.pos[0] = w - pad; self.vel[0] = -abs(self.vel[0])
        if self.pos[1] < pad:
            self.pos[1] = pad; self.vel[1] = abs(self.vel[1])
        if self.pos[1] > h - pad:
            self.pos[1] = h - pad; self.vel[1] = -abs(self.vel[1])

    def draw(self, img):
        cx, cy = int(self.pos[0]), int(self.pos[1])
        color = (0, 255, 0) if self.grabbed else (0, 100, 255)
        draw_ball(img, (cx, cy), BALL_RADIUS, color)

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

    grabbed_ball = None
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

        # result.hand_landmarks is a list of hands;
        # each hand is a list of 21 NormalizedLandmark (x, y, z in [0,1]).
        pinch_points = []   # (pinch_x, pinch_y, pinch_dist_px)
        raw_lms_list = []   # pixel coords for all 21 pts per hand

        for hand_lms in result.hand_landmarks:
            ix = int(hand_lms[TIP_INDEX].x * W)
            iy = int(hand_lms[TIP_INDEX].y * H)
            tx = int(hand_lms[TIP_THUMB].x * W)
            ty = int(hand_lms[TIP_THUMB].y * H)
            pd = dist((ix, iy), (tx, ty))
            pinch_points.append(((ix + tx) // 2, (iy + ty) // 2, pd))
            raw_lms_list.append(
                [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms]
            )

        # ── Grab / drag / release ──────────────────────────────────
        is_pinching = any(p[2] < PINCH_DIST_PX for p in pinch_points)
        pinch_pos = None
        if pinch_points:
            for p in pinch_points:
                if p[2] < PINCH_DIST_PX:
                    pinch_pos = (p[0], p[1])
                    break
            if pinch_pos is None:
                pinch_pos = (pinch_points[0][0], pinch_points[0][1])

        if is_pinching and pinch_pos:
            if grabbed_ball is None:
                d = dist(pinch_pos, ball.pos)
                if d < BALL_RADIUS + 20:
                    grabbed_ball = ball
                    grabbed_ball.grabbed = True
            if grabbed_ball:
                grabbed_ball.pos[0] = lerp(grabbed_ball.pos[0], pinch_pos[0], 0.35)
                grabbed_ball.pos[1] = lerp(grabbed_ball.pos[1], pinch_pos[1], 0.35)
        else:
            if grabbed_ball:
                grabbed_ball.grabbed = False
                grabbed_ball = None

        # ── Update ────────────────────────────────────────────────
        ball.update(dt, W, H)

        # ── Render ────────────────────────────────────────────────
        cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0, frame)

        ball.draw(frame)

        # Hand skeleton overlay
        for i, pts in enumerate(raw_lms_list):
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (180, 180, 180), 1, cv2.LINE_AA)
            for tip_idx in FINGERTIP_INDICES:
                cv2.circle(frame, pts[tip_idx], 5,
                           (255, 255, 255), -1, cv2.LINE_AA)
            if i < len(pinch_points):
                pp = pinch_points[i]
                is_p = pp[2] < PINCH_DIST_PX
                col = (0, 255, 120) if is_p else (100, 100, 255)
                cv2.circle(frame, (pp[0], pp[1]), 12, col, 2, cv2.LINE_AA)
                if is_p:
                    cv2.circle(frame, (pp[0], pp[1]), 5, col, -1, cv2.LINE_AA)

        # HUD
        cv2.rectangle(frame, (0, 0), (W, 40), (10, 10, 20), -1)
        cv2.line(frame, (0, 40), (W, 40), (80, 60, 100), 1)
        draw_text_centered(frame, "SIMPLE PICKUP",
                           (W // 2, 25), font_scale=0.75,
                           color=(200, 160, 255))
        draw_text_centered(frame,
                           "PINCH to grab and drag the ball  |  R = reset  |  Q = quit",
                           (W // 2, H - 12), font_scale=0.42,
                           color=(140, 140, 160))

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            ball.pos = [W // 2, H // 2]
            ball.vel = [random.uniform(-100, 100), random.uniform(-100, 100)]
            ball.grabbed = False
            grabbed_ball = None

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()