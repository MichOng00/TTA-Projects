"""
Session 02 — Hand Pose Visualization
=====================================================
GOAL: Visualize hand landmarks (skeleton) and fingertips detected by MediaPipe.

Builds on Session 01: Adds hand skeleton visualization to the existing hand detection.

New Concepts:
    - Drawing hand landmarks as points
    - Drawing hand connections (skeleton structure)
    - Highlighting important fingertips (thumb, index, middle, ring, pinky)
    - Visual feedback for hand tracking

Requirements:
    pip install opencv-python pygame "mediapipe>=0.10.33" numpy
"""
# pip install opencv-python pygame "mediapipe>=0.10.33" numpy
import os
import time
import urllib.request
import cv2
import mediapipe as mp
import pygame

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOpts = mp_vision.HandLandmarkerOptions

# Constants (from Session 01)
WINDOW_NAME = "Session 02: Hand Pose Visualization"
TARGET_W = 1280
TARGET_H = 720
FPS = 60

# Hand landmarks indices (NEW - Session 02)
TIP_INDEX = 8      # index finger tip
TIP_THUMB = 4      # thumb tip
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # All five fingertips

# Hand skeleton connections (NEW - Session 02)
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
        print("[Session 02] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Session 02] Model saved to", MODEL_PATH)


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

    cap = cv2.VideoCapture(0)

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

    while running:
        _, frame = cap.read()

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

        # Extract hand landmarks and draw skeleton
        for hand_idx, hand_lms in enumerate(result.hand_landmarks):
            # Convert normalized coordinates to pixel coordinates
            pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms]
            
            # Draw hand connections (skeleton)
            for a, b in HAND_CONNECTIONS:
                pygame.draw.line(screen, (180, 180, 180), pts[a], pts[b], 1)
            
            # Highlight all fingertips with larger circles
            for tip_idx in FINGERTIP_INDICES:
                pygame.draw.circle(screen, (255, 255, 100), pts[tip_idx], 6)
            
            # Highlight thumb and index with special colors
            pygame.draw.circle(screen, (255, 100, 100), pts[TIP_THUMB], 8)      # Thumb in red
            pygame.draw.circle(screen, (100, 100, 255), pts[TIP_INDEX], 8)      # Index in blue

        # Draw info box
        pygame.draw.rect(screen, (10, 10, 20), (0, 0, W, 100))
        pygame.draw.line(screen, (80, 60, 100), (0, 100), (W, 100), 1)
        
        title_surf = font_large.render("SESSION 02: Hand Pose Visualization", True, (200, 160, 255))
        hands_surf = font_small.render(f"Hands detected: {len(result.hand_landmarks)}", True, (160, 220, 255))
        info_surf = font_small.render("Red = Thumb  |  Blue = Index  |  Yellow = All Fingertips  |  Q = quit", True, (140, 140, 160))
        
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
