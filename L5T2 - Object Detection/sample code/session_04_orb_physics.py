"""
Session 04 — Orb Physics & Pedestals
=====================================================
GOAL: Create game objects (Orbs and Pedestals) with physics, glowing effects, and visual polish.

Builds on Session 03: Adds game objects while keeping hand detection and pinch visualization.

New Concepts:
    - Object-oriented design with Orb and Pedestal classes
    - Physics simulation (velocity, bouncing, friction)
    - Custom drawing functions for visual effects
    - Glow/light effects for game objects
    - Color management for different object types
    - Text rendering with shadows
    - Game object initialization

Requirements:
    pip install opencv-python pygame "mediapipe>=0.10.33" numpy
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

# Constants
WINDOW_NAME = "Session 04: Orb Physics & Pedestals"
TARGET_W = 1280
TARGET_H = 720
FPS = 60

# Hand landmarks indices
TIP_INDEX = 8
TIP_THUMB = 4
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# Game constants
ORB_RADIUS = 38
PEDESTAL_RADIUS = 52

# Color definitions
COLORS = {
    "fire":   (45, 100, 255),      # Blue-ish fire
    "ice":    (230, 210, 80),      # Cyan ice
    "nature": (50, 200, 80),       # Green nature
    "shadow": (170, 50, 180),      # Purple shadow
    "light":  (50, 220, 230),      # Yellow light
}
COLOR_NAMES = list(COLORS.keys())
NUM_ORBS = len(COLOR_NAMES)

# Model download
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[Session 04] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Session 04] Model saved to", MODEL_PATH)


def draw_glow_circle(surface, center, radius, color, layers=4):
    """Draw a glowing circle with layers of glow and a bright core."""
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    
    # Draw glow layers
    for i in range(layers, 0, -1):
        glow_color = (*color, int(60 / i))
        pygame.draw.circle(overlay, glow_color, center, radius + i * 7)
    
    surface.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    
    # Draw main orb
    pygame.draw.circle(surface, color, center, radius)
    
    # Draw bright core
    core = tuple(min(255, int(c * 1.6)) for c in color)
    pygame.draw.circle(surface, core, center, radius // 3)


def draw_pedestal(surface, center, radius, color, filled=False):
    """Draw a pedestal/target platform with decorative lines."""
    cx, cy = center
    
    # Outer ring
    pygame.draw.circle(surface, color, center, radius + 6, 2)
    
    # Inner ring
    pygame.draw.circle(surface, color, center, radius - 4, 1)
    
    # Decorative spikes
    for i in range(8):
        a = math.radians(i * 45)
        x1 = int(cx + (radius - 10) * math.cos(a))
        y1 = int(cy + (radius - 10) * math.sin(a))
        x2 = int(cx + (radius + 2) * math.cos(a))
        y2 = int(cy + (radius + 2) * math.sin(a))
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)
    
    # Fill if active
    if filled:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (*color, 60), center, radius)
        surface.blit(overlay, (0, 0))


def draw_text_centered(surface, text, center, font, color=(255, 255, 255)):
    """Draw text centered at a position with shadow."""
    text_surf = font.render(text, True, color)
    shadow_surf = font.render(text, True, (0, 0, 0))
    rect = text_surf.get_rect(center=center)
    surface.blit(shadow_surf, rect.move(1, 1))
    surface.blit(text_surf, rect)


class Orb:
    """A colored sphere that bounces around the screen."""
    def __init__(self, name, color, pos):
        self.name = name
        self.color = color
        self.pos = list(pos)
        self.vel = [random.uniform(-40, 40), random.uniform(-40, 40)]
        self.wobble = random.uniform(0, 2 * math.pi)

    def update(self, dt, w, h):
        """Update orb position with physics simulation."""
        self.wobble += dt * 1.8
        
        # Apply velocity
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        
        # Bounce off walls
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
        """Draw the orb with wobble animation."""
        cx = int(self.pos[0])
        cy = int(self.pos[1]) + int(math.sin(self.wobble) * 5)
        draw_glow_circle(surface, (cx, cy), ORB_RADIUS, self.color)
        draw_text_centered(surface, self.name[:3].upper(), (cx, cy), pygame.font.Font(None, 24))


class Pedestal:
    """A target platform where orbs can be placed."""
    def __init__(self, name, color, pos):
        self.name = name
        self.color = color
        self.pos = pos
        self.filled = False
        self.anim_t = 0.0

    def draw(self, surface, dt):
        """Draw the pedestal with animation."""
        cx, cy = self.pos
        draw_pedestal(surface, (cx, cy), PEDESTAL_RADIUS, self.color, filled=self.filled)
        draw_text_centered(surface, self.name[:3].upper(),
                           (cx, cy + PEDESTAL_RADIUS + 16), pygame.font.Font(None, 24),
                           color=self.color)
        
        # Animate when filled
        if self.filled:
            self.anim_t += dt
            pulse_r = PEDESTAL_RADIUS + int(8 * abs(math.sin(self.anim_t * 3)))
            bright = tuple(min(255, int(c * 1.4)) for c in self.color)
            pygame.draw.circle(surface, bright, (cx, cy), pulse_r, 1)


def make_game(w, h):
    """Initialize game objects."""
    pedestal_y = h - 80
    xs = np.linspace(w * 0.15, w * 0.85, NUM_ORBS).astype(int)
    
    # Shuffle pedestal colors
    names = COLOR_NAMES[:]
    random.shuffle(names)
    
    # Create pedestals at bottom
    pedestals = [Pedestal(n, COLORS[n], (int(xs[i]), pedestal_y))
                 for i, n in enumerate(names)]
    
    # Create orbs scattered across screen
    orbs = [Orb(n, COLORS[n],
                (random.randint(int(w * 0.1), int(w * 0.9)),
                 random.randint(100, int(h * 0.55))))
            for n in COLOR_NAMES]
    
    return orbs, pedestals


def run():
    """Main application loop."""
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

    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Session 04] ERROR: Unable to open the camera.")
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
    
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        now = time.time()
        timestamp_ms = int(now * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surf = pygame.image.frombuffer(rgb.tobytes(), (W, H), "RGB")
        screen.blit(frame_surf, (0, 0))

        # Dim overlay
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        # Run hand detection (from Session 01-03)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Extract and visualize hand landmarks (from Session 02-03)
        raw_lms_list = []
        for hand_lms in result.hand_landmarks:
            pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms]
            raw_lms_list.append(pts)
            
            # Draw hand connections (skeleton)
            for a, b in HAND_CONNECTIONS:
                pygame.draw.line(screen, (180, 180, 180), pts[a], pts[b], 1)
            
            # Highlight fingertips
            for tip_idx in FINGERTIP_INDICES:
                pygame.draw.circle(screen, (255, 255, 100), pts[tip_idx], 6)
            
            # Highlight thumb and index
            pygame.draw.circle(screen, (255, 100, 100), pts[TIP_THUMB], 8)
            pygame.draw.circle(screen, (100, 100, 255), pts[TIP_INDEX], 8)

        # Update and draw game objects
        for orb in orbs:
            orb.update(dt, W, H)
            orb.draw(screen)
        
        for ped in pedestals:
            ped.draw(screen, dt)

        # Draw info box
        pygame.draw.rect(screen, (10, 10, 20), (0, 0, W, 50))
        pygame.draw.line(screen, (80, 60, 100), (0, 50), (W, 50), 1)
        draw_text_centered(screen, "SESSION 04: Orb Physics & Pedestals", (W // 2, 25), font_large, (200, 160, 255))
        
        # Instructions
        draw_text_centered(screen, "Orbs bounce with physics | Pedestals wait for matches | Q = quit",
                           (W // 2, H - 12), font_small, (140, 140, 160))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        pygame.display.flip()

    cap.release()
    landmarker.close()
    pygame.quit()


if __name__ == "__main__":
    run()
