"""
Mystic Shelf — A hand-tracking grab game
=========================================
Use your hand to PINCH and DRAG glowing orbs into their matching pedestals.

Requirements:
    pip install opencv-python "mediapipe>=0.10.33" numpy

    The script auto-downloads the hand_landmarker.task model (~8 MB) on first
    run and caches it next to this script as  hand_landmarker.task

Controls:
    Q / ESC  — Quit
    R        — Restart / shuffle orbs
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
WINDOW_NAME     = "Mystic Shelf"
TARGET_W        = 1280
TARGET_H        = 720

ORB_RADIUS      = 38
PEDESTAL_RADIUS = 52
PINCH_DIST_PX   = 60
SNAP_DIST_PX    = 70
PARTICLE_LIFE   = 0.6

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

COLORS = {
    "fire":   ( 45, 100, 255),
    "ice":    (230, 210,  80),
    "nature": ( 50, 200,  80),
    "shadow": (170,  50, 180),
    "light":  ( 50, 220, 230),
}
COLOR_NAMES = list(COLORS.keys())
NUM_ORBS    = len(COLOR_NAMES)

# ──────────────────────────────────────────────
#  Model bootstrap
# ──────────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[Mystic Shelf] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Mystic Shelf] Model saved to", MODEL_PATH)

# ──────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def lerp(a, b, t):
    return a + (b - a) * t

def draw_glow_circle(img, center, radius, color, layers=4):
    overlay = img.copy()
    for i in range(layers, 0, -1):
        cv2.circle(overlay, center, int(radius + i * 7), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    core = tuple(min(255, int(c * 1.6)) for c in color)
    cv2.circle(img, center, radius // 3, core, -1, cv2.LINE_AA)

def draw_pedestal(img, center, radius, color, filled=False):
    cx, cy = center
    cv2.circle(img, (cx, cy), radius + 6, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius - 4, color, 1, cv2.LINE_AA)
    for i in range(8):
        a = math.radians(i * 45)
        x1 = int(cx + (radius - 10) * math.cos(a))
        y1 = int(cy + (radius - 10) * math.sin(a))
        x2 = int(cx + (radius + 2)  * math.cos(a))
        y2 = int(cy + (radius + 2)  * math.sin(a))
        cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    if filled:
        ov = img.copy()
        cv2.circle(ov, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)

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
#  Particle system
# ──────────────────────────────────────────────

class Particle:
    def __init__(self, pos, color):
        self.x, self.y = float(pos[0]), float(pos[1])
        angle  = random.uniform(0, 2 * math.pi)
        speed  = random.uniform(40, 140)
        self.vx     = math.cos(angle) * speed
        self.vy     = math.sin(angle) * speed
        self.color  = color
        self.born   = time.time()
        self.life   = PARTICLE_LIFE + random.uniform(-0.1, 0.1)
        self.radius = random.randint(3, 7)

    def update(self, dt):
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.vy += 120 * dt

    def alive(self):
        return (time.time() - self.born) < self.life

    def draw(self, img):
        age   = (time.time() - self.born) / self.life
        alpha = max(0.0, 1.0 - age)
        r     = max(1, int(self.radius * (1 - age * 0.5)))
        ov    = img.copy()
        cv2.circle(ov, (int(self.x), int(self.y)), r,
                   self.color, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha * 0.9, img, 1 - alpha * 0.9, 0, img)

# ──────────────────────────────────────────────
#  Game objects
# ──────────────────────────────────────────────

class Orb:
    def __init__(self, name, color, pos):
        self.name    = name
        self.color   = color
        self.pos     = list(pos)
        self.vel     = [random.uniform(-40, 40), random.uniform(-40, 40)]
        self.grabbed = False
        self.placed  = False
        self.wobble  = random.uniform(0, 2 * math.pi)

    def update(self, dt, w, h):
        if self.grabbed or self.placed:
            return
        self.wobble  += dt * 1.8
        self.pos[0]  += self.vel[0] * dt
        self.pos[1]  += self.vel[1] * dt
        pad = ORB_RADIUS + 10
        if self.pos[0] < pad:
            self.pos[0] = pad;      self.vel[0] =  abs(self.vel[0])
        if self.pos[0] > w - pad:
            self.pos[0] = w - pad;  self.vel[0] = -abs(self.vel[0])
        if self.pos[1] < pad + 60:
            self.pos[1] = pad + 60; self.vel[1] =  abs(self.vel[1])
        if self.pos[1] > h - pad - 80:
            self.pos[1] = h - pad - 80; self.vel[1] = -abs(self.vel[1])

    def draw(self, img):
        if self.placed:
            return
        cx = int(self.pos[0])
        cy = int(self.pos[1]) + (0 if self.grabbed
                                  else int(math.sin(self.wobble) * 5))
        draw_glow_circle(img, (cx, cy), ORB_RADIUS, self.color)
        draw_text_centered(img, self.name[:3].upper(), (cx, cy),
                           font_scale=0.45)


class Pedestal:
    def __init__(self, name, color, pos):
        self.name   = name
        self.color  = color
        self.pos    = pos
        self.filled = False
        self.anim_t = 0.0

    def draw(self, img, dt):
        cx, cy = self.pos
        draw_pedestal(img, (cx, cy), PEDESTAL_RADIUS,
                      self.color, filled=self.filled)
        draw_text_centered(img, self.name[:3].upper(),
                           (cx, cy + PEDESTAL_RADIUS + 16),
                           font_scale=0.45, color=self.color)
        if self.filled:
            self.anim_t += dt
            pulse_r = PEDESTAL_RADIUS + int(8 * abs(math.sin(self.anim_t * 3)))
            bright  = tuple(min(255, int(c * 1.4)) for c in self.color)
            cv2.circle(img, (cx, cy), pulse_r, bright, 1, cv2.LINE_AA)

# ──────────────────────────────────────────────
#  Game factory
# ──────────────────────────────────────────────

def make_game(w, h):
    pedestal_y = h - 80
    xs = np.linspace(w * 0.15, w * 0.85, NUM_ORBS).astype(int)
    names = COLOR_NAMES[:]
    random.shuffle(names)
    pedestals = [Pedestal(n, COLORS[n], (int(xs[i]), pedestal_y))
                 for i, n in enumerate(names)]
    orbs = [Orb(n, COLORS[n],
                (random.randint(int(w * 0.1), int(w * 0.9)),
                 random.randint(100, int(h * 0.55))))
            for n in COLOR_NAMES]
    return orbs, pedestals

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

    cap = None
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Mystic Shelf] ERROR: Unable to open the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    orbs, pedestals = make_game(W, H)
    particles: list[Particle] = []

    grabbed_orb: Orb | None  = None
    score      = 0
    start_time = time.time()
    prev_time  = start_time
    win_time: float | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Mystic Shelf] ERROR: Camera frame read failed.")
            break

        frame        = cv2.flip(frame, 1)
        now          = time.time()
        dt           = min(now - prev_time, 0.05)
        prev_time    = now
        timestamp_ms = int(now * 1000)

        # ── Hand detection via Tasks API ───────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect_for_video(mp_image, timestamp_ms)

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

        # ── Grab / drag / release ──────────────────────────────────
        is_pinching = any(p[2] < PINCH_DIST_PX for p in pinch_points)
        pinch_pos   = None
        if pinch_points:
            for p in pinch_points:
                if p[2] < PINCH_DIST_PX:
                    pinch_pos = (p[0], p[1])
                    break
            if pinch_pos is None:
                pinch_pos = (pinch_points[0][0], pinch_points[0][1])

        if is_pinching and pinch_pos:
            if grabbed_orb is None:
                best_d, best_orb = PINCH_DIST_PX * 1.8, None
                for orb in orbs:
                    if not orb.placed:
                        d = dist(pinch_pos, orb.pos)
                        if d < best_d:
                            best_d, best_orb = d, orb
                if best_orb:
                    grabbed_orb = best_orb
                    grabbed_orb.grabbed = True
            if grabbed_orb:
                grabbed_orb.pos[0] = lerp(grabbed_orb.pos[0], pinch_pos[0], 0.35)
                grabbed_orb.pos[1] = lerp(grabbed_orb.pos[1], pinch_pos[1], 0.35)
        else:
            if grabbed_orb:
                for ped in pedestals:
                    if not ped.filled and ped.name == grabbed_orb.name:
                        if dist(grabbed_orb.pos, ped.pos) < SNAP_DIST_PX:
                            grabbed_orb.placed = True
                            grabbed_orb.pos    = list(ped.pos)
                            ped.filled         = True
                            score             += 1
                            for _ in range(28):
                                particles.append(
                                    Particle(ped.pos, grabbed_orb.color))
                grabbed_orb.grabbed = False
                grabbed_orb = None

        if score == NUM_ORBS and win_time is None:
            win_time = now

        # ── Update ────────────────────────────────────────────────
        for orb in orbs:
            orb.update(dt, W, H)
        for p in particles:
            p.update(dt)
        particles = [p for p in particles if p.alive()]

        # ── Render ────────────────────────────────────────────────
        cv2.addWeighted(frame, 0.55, np.zeros_like(frame), 0.45, 0, frame)
        for y in range(0, H, 4):
            cv2.line(frame, (0, y), (W, y), (0, 0, 0), 1)
        cv2.addWeighted(frame, 0.88, np.zeros_like(frame), 0.12, 0, frame)

        for ped in pedestals:
            ped.draw(frame, dt)

        for orb in orbs:
            orb.draw(frame)

        for orb in orbs:
            if orb.placed:
                cx, cy = int(orb.pos[0]), int(orb.pos[1])
                draw_glow_circle(frame, (cx, cy), ORB_RADIUS - 6, orb.color)

        for p in particles:
            p.draw(frame)

        # Hand skeleton overlay
        for i, pts in enumerate(raw_lms_list):
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (180, 180, 180), 1, cv2.LINE_AA)
            for tip_idx in FINGERTIP_INDICES:
                cv2.circle(frame, pts[tip_idx], 5,
                           (255, 255, 255), -1, cv2.LINE_AA)
            if i < len(pinch_points):
                pp   = pinch_points[i]
                is_p = pp[2] < PINCH_DIST_PX
                col  = (0, 255, 120) if is_p else (100, 100, 255)
                cv2.circle(frame, (pp[0], pp[1]), 12, col, 2, cv2.LINE_AA)
                if is_p:
                    cv2.circle(frame, (pp[0], pp[1]), 5, col, -1, cv2.LINE_AA)

        # HUD
        cv2.rectangle(frame, (0, 0), (W, 50), (10, 10, 20), -1)
        cv2.line(frame, (0, 50), (W, 50), (80, 60, 100), 1)
        elapsed    = int(now - start_time) if win_time is None \
                     else int(win_time - start_time)
        mins, secs = divmod(elapsed, 60)
        draw_text_centered(frame, "MYSTIC SHELF",
                           (W // 2, 25), font_scale=0.75,
                           color=(200, 160, 255))
        draw_text_centered(frame, f"PLACED  {score} / {NUM_ORBS}",
                           (W // 4, 25), font_scale=0.6,
                           color=(160, 220, 255))
        draw_text_centered(frame, f"{mins:02d}:{secs:02d}",
                           (W * 3 // 4, 25), font_scale=0.6,
                           color=(160, 220, 255))
        draw_text_centered(frame,
                           "PINCH to grab  |  R = restart  |  Q = quit",
                           (W // 2, H - 12), font_scale=0.42,
                           color=(140, 140, 160))

        if win_time is not None:
            flash = abs(math.sin((now - win_time) * 3))
            col   = tuple(int(c * flash) for c in (80, 255, 180))
            draw_text_centered(frame,
                               f"YOU WIN!  Time: {mins:02d}:{secs:02d}",
                               (W // 2, H // 2),
                               font_scale=1.4, color=col, thickness=2)
            draw_text_centered(frame, "Press R to play again",
                               (W // 2, H // 2 + 55),
                               font_scale=0.65, color=(200, 200, 200))

        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            orbs, pedestals = make_game(W, H)
            particles.clear()
            grabbed_orb = None
            score       = 0
            start_time  = time.time()
            win_time    = None

        cv2.imshow(WINDOW_NAME, frame)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()