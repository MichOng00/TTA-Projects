import cv2
import time
import math
import mediapipe as mp
import random
from dataclasses import dataclass

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions

@dataclass
class Note:
    lane: int
    y: float
    speed: float = 30.0
    active: bool = True

    def update(self, dt):
        self.y += self.speed * dt
        if self.y > 540:
            self.active = False

class RhythmGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        model_path = "hand_landmarker.task"
        base_options = BaseOptions(model_asset_path = model_path)
        options = HandLandmarkerOptions(
            base_options = base_options,
            running_mode = mp_vision.RunningMode.VIDEO,
            num_hands = 1
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.lane_count = 3
        self.lane_x = [int((i + 0.5) * self.frame_width / self.lane_count) for i in range(self.lane_count)]

        self.hit_zone_y = int(self.frame_height * 0.8)
        self.hit_window = 80 # how many px around hit zone to count

        self.spawn_interval = 3 # seconds
        self.last_spawn_time = time.time()

    def run(self):
        while True:
            ret, frame = self.cap.read()

            current_time = time.time()

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # run hand detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect_for_video(mp_image, int(current_time*1000))

            hand_info = self.extract_hand_info(results, frame.shape)
            
            self.draw_game(frame, hand_info)

            cv2.imshow("Rhythm game", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def extract_hand_info(self, results, shape):
        hand_info = {"found": False, "x": 0, "y": 0, "pinch": False}
        if not results.hand_landmarks:
            return hand_info
        
        hand_lms = results.hand_landmarks[0]
        width, height = shape[1], shape[0]

        index_tip = hand_lms[8]
        thumb_tip = hand_lms[4]

        # pixel coordinates
        px = int(index_tip.x * width)
        py = int(index_tip.y * height)

        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        pinch = pinch_dist < 0.055 # threshold

        hand_info.update({"found": True, "x":px, "y": py, "pinch":pinch})
        return hand_info
    
    def draw_game(self, frame, hand_info):
        lane_color = (180, 180, 180)

        if hand_info["found"]:
            hand_color = (220, 80, 80) if hand_info["pinch"] else (80, 190, 220)
            cv2.circle(frame, (hand_info["x"], hand_info["y"]), 16, hand_color, -1)

            cv2.putText(frame, f"X:{hand_info['x']}, Y:{hand_info['y']}", 
                        (hand_info["x"] - 80, hand_info["y"] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

# if __name__ == "__main__":
game = RhythmGame()
game.run()