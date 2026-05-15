"""
Mystic Shelf — Pygame version
=========================================
Use your hand to PINCH and DRAG glowing orbs into their matching pedestals.

Requirements:
    pip install opencv-python pygame "mediapipe>=0.10.33" numpy

The script auto-downloads the hand_landmarker.task model (~8 MB) on first
run and caches it next to this script as hand_landmarker.task.

Controls:
    Q / ESC  — Quit
    R        — Restart / shuffle orbs
"""

import os
import math
import random
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pygame

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOpts = mp_vision.HandLandmarkerOptions

TIP_INDEX = 8   # index finger tip
TIP_THUMB = 4   # thumb tip

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

WINDOW_NAME = "Mystic Shelf"
TARGET_W = 1280
TARGET_H = 720
FPS = 60

ORB_RADIUS = 38
PEDESTAL_RADIUS = 52
PINCH_DIST_PX = 60
SNAP_DIST_PX = 70
PARTICLE_LIFE = 0.6

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

COLORS = {
    "fire":   (45, 100, 255),
    "ice":    (230, 210, 80),
    "nature": (50, 200, 80),
    "shadow": (170, 50, 180),
    "light":  (50, 220, 230),
}
COLOR_NAMES = list(COLORS.keys())
NUM_ORBS = len(COLOR_NAMES)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[Mystic Shelf] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Mystic Shelf] Model saved to", MODEL_PATH)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a, b, t):
    return a + (b - a) * t


def draw_glow_circle(surface, center, radius, color, layers=4):
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        glow_color = (*color, int(60 / i))
        pygame.draw.circle(overlay, glow_color, center,
                           radius + i * 7)
    surface.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    pygame.draw.circle(surface, color, center, radius)
    core = tuple(min(255, int(c * 1.6)) for c in color)
    pygame.draw.circle(surface, core, center, radius // 3)


def draw_pedestal(surface, center, radius, color, filled=False):
    cx, cy = center
    pygame.draw.circle(surface, color, center, radius + 6, 2)
    pygame.draw.circle(surface, color, center, radius - 4, 1)
    for i in range(8):
        a = math.radians(i * 45)
        x1 = int(cx + (radius - 10) * math.cos(a))
        y1 = int(cy + (radius - 10) * math.sin(a))
        x2 = int(cx + (radius + 2) * math.cos(a))
        y2 = int(cy + (radius + 2) * math.sin(a))
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)
    if filled:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (*color, 60), center, radius)
        surface.blit(overlay, (0, 0))


def draw_text_centered(surface, text, center, font, color=(255, 255, 255)):
    text_surf = font.render(text, True, color)
    shadow_surf = font.render(text, True, (0, 0, 0))
    rect = text_surf.get_rect(center=center)
    surface.blit(shadow_surf, rect.move(1, 1))
    surface.blit(text_surf, rect)


class Particle:
    def __init__(self, pos, color):
        self.x, self.y = float(pos[0]), float(pos[1])
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(40, 140)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.born = time.time()
        self.life = PARTICLE_LIFE + random.uniform(-0.1, 0.1)
        self.radius = random.randint(3, 7)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 120 * dt

    def alive(self):
        return (time.time() - self.born) < self.life

    def draw(self, surface):
        age = (time.time() - self.born) / self.life
        alpha = max(0.0, 1.0 - age)
        radius = max(1, int(self.radius * (1 - age * 0.5)))
        overlay = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (*self.color, int(alpha * 230)),
                           (radius + 2, radius + 2), radius)
        surface.blit(overlay, (int(self.x) - radius - 2, int(self.y) - radius - 2))


class Orb:
    def __init__(self, name, color, pos):
        self.name = name
        self.color = color
        self.pos = list(pos)
        self.vel = [random.uniform(-40, 40), random.uniform(-40, 40)]
        self.grabbed = False
        self.placed = False
        self.wobble = random.uniform(0, 2 * math.pi)

    def update(self, dt, w, h):
        if self.grabbed or self.placed:
            return
        self.wobble += dt * 1.8
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        pad = ORB_RADIUS + 10
        if self.pos[0] < pad:
            self.pos[0] = pad
            self.vel[0] = abs(self.vel[0])
        if self.pos[0] > w - pad:
            self.pos[0] = w - pad
            self.vel[0] = -abs(self.vel[0])
        if self.pos[1] < pad + 60:
            self.pos[1] = pad + 60
            self.vel[1] = abs(self.vel[1])
        if self.pos[1] > h - pad - 80:
            self.pos[1] = h - pad - 80
            self.vel[1] = -abs(self.vel[1])

    def draw(self, surface):
        if self.placed:
            return
        cx = int(self.pos[0])
        cy = int(self.pos[1]) + (0 if self.grabbed else int(math.sin(self.wobble) * 5))
        draw_glow_circle(surface, (cx, cy), ORB_RADIUS, self.color)
        draw_text_centered(surface, self.name[:3].upper(), (cx, cy), pygame.font.Font(None, 24))


class Pedestal:
    def __init__(self, name, color, pos):
        self.name = name
        self.color = color
        self.pos = pos
        self.filled = False
        self.anim_t = 0.0

    def draw(self, surface, dt):
        cx, cy = self.pos
        draw_pedestal(surface, (cx, cy), PEDESTAL_RADIUS, self.color, filled=self.filled)
        draw_text_centered(surface, self.name[:3].upper(),
                           (cx, cy + PEDESTAL_RADIUS + 16), pygame.font.Font(None, 24),
                           color=self.color)
        if self.filled:
            self.anim_t += dt
            pulse_r = PEDESTAL_RADIUS + int(8 * abs(math.sin(self.anim_t * 3)))
            bright = tuple(min(255, int(c * 1.4)) for c in self.color)
            pygame.draw.circle(surface, bright, (cx, cy), pulse_r, 1)


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


def run():
    ensure_model()

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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(WINDOW_NAME)
    clock = pygame.time.Clock()
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 30)
    font_small = pygame.font.Font(None, 22)

    orbs, pedestals = make_game(W, H)
    particles = []

    grabbed_orb = None
    score = 0
    start_time = time.time()
    win_time = None

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            print("[Mystic Shelf] ERROR: Camera frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        now = time.time()
        timestamp_ms = int(now * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surf = pygame.image.frombuffer(rgb.tobytes(), (W, H), "RGB")
        screen.blit(frame_surf, (0, 0))

        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))
        for y in range(0, H, 4):
            pygame.draw.line(screen, (0, 0, 0, 20), (0, y), (W, y), 1)
        screen.blit(overlay, (0, 0))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        pinch_points = []
        raw_lms_list = []
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
                            grabbed_orb.pos = list(ped.pos)
                            ped.filled = True
                            score += 1
                            for _ in range(28):
                                particles.append(Particle(ped.pos, grabbed_orb.color))
                grabbed_orb.grabbed = False
                grabbed_orb = None

        if score == NUM_ORBS and win_time is None:
            win_time = now

        for orb in orbs:
            orb.update(dt, W, H)
        for p in particles:
            p.update(dt)
        particles = [p for p in particles if p.alive()]

        for ped in pedestals:
            ped.draw(screen, dt)
        for orb in orbs:
            orb.draw(screen)
        for orb in orbs:
            if orb.placed:
                cx, cy = int(orb.pos[0]), int(orb.pos[1])
                draw_glow_circle(screen, (cx, cy), ORB_RADIUS - 6, orb.color)
        for p in particles:
            p.draw(screen)

        for pts in raw_lms_list:
            for a, b in HAND_CONNECTIONS:
                pygame.draw.line(screen, (180, 180, 180), pts[a], pts[b], 1)
            for tip_idx in FINGERTIP_INDICES:
                pygame.draw.circle(screen, (255, 255, 255), pts[tip_idx], 5)
        for i, pp in enumerate(pinch_points):
            is_p = pp[2] < PINCH_DIST_PX
            col = (0, 255, 120) if is_p else (100, 100, 255)
            pygame.draw.circle(screen, col, (pp[0], pp[1]), 12, 2)
            if is_p:
                pygame.draw.circle(screen, col, (pp[0], pp[1]), 5)

        pygame.draw.rect(screen, (10, 10, 20), (0, 0, W, 50))
        pygame.draw.line(screen, (80, 60, 100), (0, 50), (W, 50), 1)
        elapsed = int(now - start_time) if win_time is None else int(win_time - start_time)
        mins, secs = divmod(elapsed, 60)
        draw_text_centered(screen, "MYSTIC SHELF", (W // 2, 25), font_large, (200, 160, 255))
        draw_text_centered(screen, f"PLACED  {score} / {NUM_ORBS}", (W // 4, 25), font_medium, (160, 220, 255))
        draw_text_centered(screen, f"{mins:02d}:{secs:02d}", (W * 3 // 4, 25), font_medium, (160, 220, 255))
        draw_text_centered(screen, "PINCH to grab  |  R = restart  |  Q = quit",
                           (W // 2, H - 12), font_small, (140, 140, 160))

        if win_time is not None:
            flash = abs(math.sin((now - win_time) * 3))
            col = tuple(int(c * flash) for c in (80, 255, 180))
            draw_text_centered(screen, f"YOU WIN!  Time: {mins:02d}:{secs:02d}",
                               (W // 2, H // 2), font_large, col)
            draw_text_centered(screen, "Press R to play again",
                               (W // 2, H // 2 + 55), font_medium, (200, 200, 200))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    orbs, pedestals = make_game(W, H)
                    particles.clear()
                    grabbed_orb = None
                    score = 0
                    start_time = time.time()
                    win_time = None

        pygame.display.flip()

    cap.release()
    landmarker.close()
    pygame.quit()


if __name__ == "__main__":
    run()
