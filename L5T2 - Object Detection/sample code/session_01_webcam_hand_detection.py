"""
Session 01 — Webcam + Hand Detection
=====================================================
GOAL: Capture video from webcam and detect hand positions using MediaPipe.

Concepts:
    - OpenCV camera capture
    - MediaPipe HandLandmarker
    - Converting between OpenCV and MediaPipe image formats
    - Drawing on frames
    - Basic PyGame display

Requirements:
    pip install opencv-python pygame "mediapipe>=0.10.33" numpy
"""

import os
import math
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
WINDOW_NAME = "Session 01: Webcam + Hand Detection"
TARGET_W = 1280
TARGET_H = 720
FPS = 60

# Hand landmarks indices
TIP_INDEX = 8   # index finger tip
TIP_THUMB = 4   # thumb tip

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
        print("[Session 01] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Session 01] Model saved to", MODEL_PATH)


def run():
    """Main application loop."""
    ensure_model()

    # Create the hand landmarker
    options = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Open camera
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Session 01] ERROR: Unable to open the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(WINDOW_NAME)
    clock = pygame.time.Clock()
    font_small = pygame.font.Font(None, 22)
    font_large = pygame.font.Font(None, 48)

    running = True
    hand_count = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[Session 01] ERROR: Camera frame read failed.")
            break

        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        clock.tick(FPS)
        
        # Get current time in milliseconds for MediaPipe
        now = time.time()
        timestamp_ms = int(now * 1000)

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create pygame surface from RGB frame
        frame_surf = pygame.image.frombuffer(rgb.tobytes(), (W, H), "RGB")
        screen.blit(frame_surf, (0, 0))

        # Run hand detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Count and display detected hands
        hand_count = len(result.hand_landmarks)

        # Draw info box
        pygame.draw.rect(screen, (10, 10, 20), (0, 0, W, 100))
        pygame.draw.line(screen, (80, 60, 100), (0, 100), (W, 100), 1)
        
        title_surf = font_large.render("SESSION 01: Webcam + Hand Detection", True, (200, 160, 255))
        hands_surf = font_small.render(f"Hands detected: {hand_count}", True, (160, 220, 255))
        info_surf = font_small.render("Press Q to quit", True, (140, 140, 160))
        
        screen.blit(title_surf, (20, 10))
        screen.blit(hands_surf, (20, 50))
        screen.blit(info_surf, (20, 75))

        # Handle events
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
