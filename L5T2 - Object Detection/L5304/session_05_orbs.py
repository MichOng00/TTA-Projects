# pip install opencv-python pygame "mediapipe>=0.10.33" numpy
import os
import time
import urllib.request
import cv2
import mediapipe as mp
import pygame
import math
import random
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOpts = mp_vision.HandLandmarkerOptions

# Constants
WINDOW_NAME = "Pygame Mediapipe Hands"
TARGET_W = 1280
TARGET_H = 720
FPS = 60

ORB_RADIUS = 20
PEDESTAL_RADIUS = 40

COLORS = {
    "red" : (181, 36, 104),
    "blue" : (64, 91, 227),
    "green" : (14, 179, 64),
    "gold" : (194, 147, 6)
}
COLOR_NAMES = list(COLORS.keys())
NUM_ORBS = len(COLOR_NAMES)

# Hand landmarks indices
TIP_INDEX = 8      # index finger tip
TIP_THUMB = 4      # thumb tip
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # All five fingertips

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (5,9),(9,10),(10,11),(11,12),   # Middle
    (9,13),(13,14),(14,15),(15,16), # Ring
    (13,17),(17,18),(18,19),(19,20),# Pinky
    (0,17),                           # Palm connection
]

# Model download
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

def ensure_model():
    """Download the hand landmarker model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model saved to", MODEL_PATH)

class Orb:
    def __init__(self, name, color, pos):
        self.name = name
        self.color = color
        self.pos = list(pos)
        self.vel = [random.uniform(-40, 40), random.uniform(-40, 40)]
        self.grabbed = False

    def update(self, dt, w, h):
        if self.grabbed:
            return
        
        # moving according velocity
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

        # bounce off walls
        pad = ORB_RADIUS + 10
        if self.pos[0] < pad:
            self.pos[0] = pad
            self.vel[0] = -self.vel[0]
        if self.pos[0] > w - pad:
            self.pos[0] = w - pad
            self.vel[0] = -self.vel[0]
        if self.pos[1] < pad:
            self.pos[1] = pad
            self.vel[1] = -self.vel[1]
        if self.pos[1] > h - pad:
            self.pos[1] = h - pad
            self.vel[1] = -self.vel[1]

    def draw(self, surface):
        cx = self.pos[0]
        cy = self.pos[1]
        pygame.draw.circle(surface, self.color, (cx, cy), ORB_RADIUS, 2)

class Pedestal:
    def __init__(self, name, color, pos):
        self.name = name
        self.color = color
        self.pos = pos
        self.filled = False
        self.anim_t = 0.0

    def draw(self, surface, dt):
        cx, cy = self.pos
        pygame.draw.circle(surface, (cx, cy), PEDESTAL_RADIUS + 6, 2)
        pygame.draw.circle(surface, (cx, cy), PEDESTAL_RADIUS - 4, 2)
        if self.filled:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(overlay, (*self.color, 60), (cx, cy), PEDESTAL_RADIUS)
            surface.blit(overlay, (0,0))

            # todo: animation

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def lerp(a, b, t): # linear interpolation
    return a + (b - a) * t

def make_game(w, h):
    orbs = [Orb(n, COLORS[n], 
                (random.randint(0, w), random.randint(0, int(h*0.5))))
                for n in COLOR_NAMES]
    
    return orbs

def run():
    ensure_model()

    options = HandLandmarkerOpts(
        base_options = BaseOptions(model_asset_path = MODEL_PATH),
        running_mode = RunningMode.VIDEO,
        num_hands = 2,
        min_hand_detection_confidence = 0.6,
        min_hand_presence_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(WINDOW_NAME)
    font = pygame.font.Font(None, 30)

    orbs = make_game(W, H)
    grabbed_orb = None

    running = True
    while running:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(rgb.tobytes(), (W,H), "RGB")
        screen.blit(frame_surface, (0, 0))

        now = time.time()
        timestamp_ms = int(now * 1000)

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
            raw_lms_list.append([(int(lm.x * W), int(lm.y * H)) for lm in hand_lms])

        # detect pinching
        is_pinching = any(p[2] < 60 for p in pinch_points)
        pinch_pos = None
        if pinch_points:
            for p in pinch_points:
                if p[2] < 60:
                    pinch_pos = (p[0], p[1])
                    break
            if pinch_pos is None:
                pinch_pos = (pinch_points[0][0], pinch_points[0][1])

        # orb interaction
        if is_pinching and pinch_pos:
            if grabbed_orb is None:
                best_d, best_orb = 120, None
                for orb in orbs:
                    d = dist(pinch_pos, orb.pos)
                    if d < best_d:
                        best_d, best_orb = d, orb
                if best_orb:
                    grabbed_orb = best_orb
                    grabbed_orb.grabbed = True
            if grabbed_orb:
                grabbed_orb.pos[0] = lerp(grabbed_orb.pos[0], pinch_pos[0], 0.35)
                grabbed_orb.pos[1] = lerp(grabbed_orb.pos[1], pinch_pos[1], 0.35)
        else: # not pinching
            if grabbed_orb:
                grabbed_orb.grabbed = False
                grabbed_orb = None

        for orb in orbs:
            orb.update(0.01, W, H)
            orb.draw(screen)

        # draw hand
        for pts in raw_lms_list:
            for a, b in HAND_CONNECTIONS:
                pygame.draw.line(screen, (200, 200, 200), pts[a], pts[b], 1)
            for tip_idx in FINGERTIP_INDICES:
                pygame.draw.circle(screen, (255, 255, 255), pts[tip_idx], 5)

        for i, p in enumerate(pinch_points):
            is_p = p[2] < 60
            col = (0, 255, 120) if is_p else (100, 100, 255)
            pygame.draw.circle(screen, col, (p[0], p[1]), 12, 0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    run()
