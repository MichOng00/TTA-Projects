# pip install opencv-python pygame "mediapipe>=0.10.33" numpy
import os
import time
import urllib.request
import cv2
import mediapipe as mp
import pygame
import math

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

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

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

    running = True
    while running:
        _, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(rgb.tobytes(), (W,H), "RGB")
        screen.blit(frame_surface, (0, 0))

        now = time.time()
        timestamp_ms = int(now * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        pinch_points = []
        is_pinching = False

        # draw skeleton
        for hand_lms in result.hand_landmarks:
            # convert normalized coordinates to pixel coordinates
            pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms]

            thumb_pos = pts[TIP_THUMB]
            index_pos = pts[TIP_INDEX]

            pinch_distance = dist(thumb_pos, index_pos)

            midpoint = ((thumb_pos[0] + index_pos[0]) // 2, (thumb_pos[1] + index_pos[1]) // 2)

            pinch_points.append({
                "midpoint":midpoint,
                "distance": pinch_distance,
                "is_pinching": pinch_distance < 60
            })

            for a, b in HAND_CONNECTIONS:
                pygame.draw.line(screen, (180, 180, 180), pts[a], pts[b], 1)

            for tip_idx in FINGERTIP_INDICES:
                pygame.draw.circle(screen, (255, 255, 100), pts[tip_idx], 6)

        is_pinching = any(p["is_pinching"] for p in pinch_points)

        # draw pinch points
        for pinch_info in pinch_points:
            midpoint = pinch_info["midpoint"]            
            distance = pinch_info["distance"]            
            is_p = pinch_info["is_pinching"]

            col = (0, 255, 100) if is_p else (100, 100, 255)
            
            if is_p:
                pygame.draw.circle(screen, col, midpoint, 12, 0)
            else:
                pygame.draw.circle(screen, col, midpoint, 12, 2)
            
            dist_text = font.render(f"{distance:.0f}px", True, col)
            screen.blit(dist_text, (midpoint[0] + 15, midpoint[1] - 10))

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
