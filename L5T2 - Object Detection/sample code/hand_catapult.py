"""
Hand Catapult — Angry Birds style launching with hand pinch
==========================================================
Pinch and drag to pull back the catapult, then release to launch!

Requirements:
    pip install opencv-python "mediapipe>=0.10.33" numpy

    The script auto-downloads the hand_landmarker.task model (~8 MB) on first
    run and caches it next to this script as hand_landmarker.task

Controls:
    Q / ESC — Quit
    R       — Reset catapult
    SPACE   — Launch (alternative to pinch release)
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

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
WINDOW_NAME = "Hand Catapult"
TARGET_W    = 1280
TARGET_H    = 720
PROJECTILE_RADIUS = 15
PINCH_DIST_PX = 60
MAX_PULL_DISTANCE = 200
GRAVITY = 500  # pixels per second squared

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
        print("[Hand Catapult] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Hand Catapult] Model saved to", MODEL_PATH)

# ──────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def draw_projectile(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, (255, 255, 255), 2, cv2.LINE_AA)

def draw_catapult(img, base_pos, arm_pos, pulling=False):
    base_x, base_y = base_pos
    arm_x, arm_y = arm_pos

    # Draw base
    cv2.rectangle(img, (base_x - 20, base_y - 10), (base_x + 20, base_y + 10),
                  (100, 100, 100), -1, cv2.LINE_AA)
    cv2.rectangle(img, (base_x - 20, base_y - 10), (base_x + 20, base_y + 10),
                  (150, 150, 150), 2, cv2.LINE_AA)

    # Draw arm
    cv2.line(img, (base_x, base_y), (arm_x, arm_y), (80, 50, 20), 8, cv2.LINE_AA)
    cv2.line(img, (base_x, base_y), (arm_x, arm_y), (120, 80, 40), 4, cv2.LINE_AA)

    # Draw projectile on arm when not launched
    if not pulling:
        proj_x = int(arm_x + (base_x - arm_x) * 0.1)
        proj_y = int(arm_y + (base_y - arm_y) * 0.1)
        draw_projectile(img, (proj_x, proj_y), PROJECTILE_RADIUS, (255, 100, 100))

def draw_trajectory(img, start_pos, velocity, dt, steps=20):
    x, y = start_pos
    vx, vy = velocity
    for i in range(steps):
        x += vx * dt
        y += vy * dt
        vy += GRAVITY * dt
        if 0 <= x < TARGET_W and 0 <= y < TARGET_H:
            cv2.circle(img, (int(x), int(y)), 2, (200, 200, 200), -1, cv2.LINE_AA)
        else:
            break

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
#  Projectile class
# ──────────────────────────────────────────────

class Projectile:
    def __init__(self, pos, velocity):
        self.pos = list(pos)
        self.vel = list(velocity)
        self.launched = False
        self.landed = False

    def launch(self, start_pos, pull_distance):
        power = min(pull_distance / MAX_PULL_DISTANCE, 1.0)
        angle = math.atan2(start_pos[1] - TARGET_H + 100, start_pos[0] - TARGET_W // 2)
        speed = 300 + power * 400  # Base speed + power bonus

        self.vel[0] = math.cos(angle) * speed
        self.vel[1] = math.sin(angle) * speed
        self.pos = list(start_pos)
        self.launched = True
        self.landed = False

    def update(self, dt, w, h):
        if not self.launched or self.landed:
            return

        self.vel[1] += GRAVITY * dt
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

        # Check boundaries
        if self.pos[0] - PROJECTILE_RADIUS < 0 or self.pos[0] + PROJECTILE_RADIUS > w:
            self.vel[0] *= -0.8  # Bounce with energy loss
            self.pos[0] = max(PROJECTILE_RADIUS, min(w - PROJECTILE_RADIUS, self.pos[0]))

        if self.pos[1] + PROJECTILE_RADIUS > h:
            self.pos[1] = h - PROJECTILE_RADIUS
            self.vel[1] *= -0.5  # Ground bounce
            self.vel[0] *= 0.9
            if abs(self.vel[1]) < 50:  # Stop when slow enough
                self.landed = True

    def draw(self, img):
        if self.launched:
            cx, cy = int(self.pos[0]), int(self.pos[1])
            color = (255, 100, 100) if not self.landed else (100, 100, 100)
            draw_projectile(img, (cx, cy), PROJECTILE_RADIUS, color)

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

    catapult_base = (W // 2, H - 50)
    catapult_arm = (W // 2, H - 150)
    projectile = Projectile(catapult_arm, [0, 0])

    pulling = False
    pull_distance = 0
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
            tx = int(hand_lms[TIP_THUMB].x  * W)
            ty = int(hand_lms[TIP_THUMB].y  * H)
            pd = dist((ix, iy), (tx, ty))
            pinch_points.append(((ix + tx) // 2, (iy + ty) // 2, pd))
            raw_lms_list.append(
                [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms]
            )

        # ── Catapult interaction ───────────────────────────────────
        is_pinching = any(p[2] < PINCH_DIST_PX for p in pinch_points)
        pinch_pos = None
        if pinch_points:
            for p in pinch_points:
                if p[2] < PINCH_DIST_PX:
                    pinch_pos = (p[0], p[1])
                    break
            if pinch_pos is None:
                pinch_pos = (pinch_points[0][0], pinch_points[0][1])

        if is_pinching and pinch_pos and not projectile.launched:
            # Calculate pull distance and direction
            dx = pinch_pos[0] - catapult_base[0]
            dy = pinch_pos[1] - catapult_base[1]
            pull_distance = min(dist(pinch_pos, catapult_base), MAX_PULL_DISTANCE)
            angle = math.atan2(dy, dx)

            # Update catapult arm position
            arm_distance = 100 + pull_distance * 0.5
            catapult_arm = (
                int(catapult_base[0] + math.cos(angle) * arm_distance),
                int(catapult_base[1] + math.sin(angle) * arm_distance)
            )
            pulling = True
        elif pulling and not is_pinching:
            # Launch!
            projectile.launch(catapult_arm, pull_distance)
            pulling = False
            pull_distance = 0
            catapult_arm = (W // 2, H - 150)

        # ── Update ────────────────────────────────────────────────
        projectile.update(dt, W, H)

        # ── Render ────────────────────────────────────────────────
        cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0, frame)

        # Draw ground
        cv2.rectangle(frame, (0, H - 20), (W, H), (50, 100, 50), -1, cv2.LINE_AA)

        draw_catapult(frame, catapult_base, catapult_arm, pulling)

        if pulling and pinch_pos:
            # Draw trajectory preview
            power = min(pull_distance / MAX_PULL_DISTANCE, 1.0)
            angle = math.atan2(catapult_arm[1] - catapult_base[1],
                             catapult_arm[0] - catapult_base[0])
            speed = 300 + power * 400
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            draw_trajectory(frame, catapult_arm, (vel_x, vel_y), 0.02)

        projectile.draw(frame)

        # Hand skeleton overlay
        for i, pts in enumerate(raw_lms_list):
            for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]:
                cv2.line(frame, pts[a], pts[b], (180, 180, 180), 1, cv2.LINE_AA)
            for tip_idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, pts[tip_idx], 5, (255, 255, 255), -1, cv2.LINE_AA)
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
        draw_text_centered(frame, "HAND CATAPULT",
                           (W // 2, 25), font_scale=0.75,
                           color=(200, 160, 255))
        status = "PULL BACK" if not pulling else f"POWER: {pull_distance:.0f}"
        if projectile.launched and not projectile.landed:
            status = "LAUNCHED!"
        elif projectile.landed:
            status = "LANDED - R to reset"
        draw_text_centered(frame, status, (W // 2, 60), font_scale=0.6, color=(160, 220, 255))
        draw_text_centered(frame,
                           "PINCH to pull catapult  |  Release to launch  |  R = reset  |  Q = quit",
                           (W // 2, H - 12), font_scale=0.42,
                           color=(140, 140, 160))

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            projectile = Projectile((W // 2, H - 150), [0, 0])
            catapult_arm = (W // 2, H - 150)
            pulling = False
            pull_distance = 0
        if key == 32:  # Spacebar
            if pulling:
                projectile.launch(catapult_arm, pull_distance)
                pulling = False
                pull_distance = 0
                catapult_arm = (W // 2, H - 150)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()