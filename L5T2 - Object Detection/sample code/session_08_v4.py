import cv2
import time
import random
import math
from dataclasses import dataclass
import numpy as np
import pygame
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions


def make_tone(freq_hz: float, duration: float = 0.2, volume: float = 0.5, sample_rate: int = 44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    audio = (wave * (2 ** 15 - 1) * volume).astype(np.int16)
    # if mixer is stereo, expand to two channels
    init = pygame.mixer.get_init()
    channels = init[2] if init else 1
    if channels == 2:
        audio = np.column_stack((audio, audio))
    sound = pygame.sndarray.make_sound(audio)
    return sound


@dataclass
class Note:
    lane: int
    y: float
    speed: float = 30.0
    active: bool = True

    def update(self, dt: float) -> None:
        self.y += self.speed * dt
        if self.y > 540:
            self.active = False


class RhythmGameWithSound:
    """Hand-pinch rhythm game that plays C3/D3/E3 when notes are hit."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        # init mediapipe hand landmarker
        model_path = "hand_landmarker.task"
        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.lane_count = 3
        self.lane_x = [int((i + 0.5) * self.frame_width / self.lane_count) for i in range(self.lane_count)]
        self.hit_zone_y = int(self.frame_height * 0.82)
        self.hit_window = 80

        self.spawn_interval = 0.8
        self.last_spawn_time = time.time()

        self.notes = []
        self.score = 0
        self.combo = 0
        self.health = 6
        self.game_over = False
        self.last_hit_feedback = ""
        self.last_hit_time = 0.0

        # init pygame mixer and synth sounds
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        # Frequencies: C3, D3, E3
        freqs = [130.81, 146.83, 164.81]
        self.sounds = [make_tone(f, duration=0.18, volume=0.6) for f in freqs]

    def extract_hand_info(self, results, shape) -> dict:
        hand_info = {"found": False, "x": 0, "y": 0, "pinch": False}
        if not results.hand_landmarks:
            return hand_info

        hand_lms = results.hand_landmarks[0]
        width, height = shape[1], shape[0]

        index_tip = hand_lms[8]
        thumb_tip = hand_lms[4]
        px = int(index_tip.x * width)
        py = int(index_tip.y * height)

        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        pinch = pinch_dist < 0.055

        hand_info.update({"found": True, "x": px, "y": py, "pinch": pinch})
        hand_info["landmarks"] = hand_lms
        return hand_info

    def spawn_notes(self, current_time):
        if current_time - self.last_spawn_time < self.spawn_interval:
            return
        lane = random.randint(0, self.lane_count - 1)
        self.notes.append(Note(lane=lane, y=-50.0, speed=40.0))
        self.last_spawn_time = current_time

    def update_notes(self, hand_info, current_time):
        for note in self.notes:
            if not note.active:
                continue
            note.update(current_time - self.last_spawn_time)

            if note.y > self.hit_zone_y + self.hit_window:
                note.active = False
                self.combo = 0
                self.health -= 1
                self.last_hit_feedback = "Miss"
                self.last_hit_time = current_time
                continue

            if hand_info["found"] and hand_info["pinch"] and self.is_note_in_lane(note, hand_info):
                distance = abs(note.y - self.hit_zone_y)
                if distance < self.hit_window:
                    note.active = False
                    points = 120 if distance < 30 else 90 if distance < 55 else 60
                    self.score += points + self.combo * 5
                    self.combo += 1
                    self.last_hit_feedback = "Perfect" if distance < 30 else "Great" if distance < 55 else "Good"
                    self.last_hit_time = current_time
                    # play sound for lane
                    try:
                        self.sounds[note.lane].play()
                    except Exception:
                        pass

        self.notes = [note for note in self.notes if note.active]
        if self.health <= 0:
            self.game_over = True

    def is_note_in_lane(self, note: Note, hand_info: dict) -> bool:
        lane_center = self.lane_x[note.lane]
        return abs(hand_info["x"] - lane_center) < self.frame_width // (self.lane_count * 2)

    def draw_game(self, frame, hand_info, current_time):
        lane_color = (180, 180, 180)
        active_color = (255, 255, 255)

        for x in self.lane_x:
            cv2.line(frame, (x, 0), (x, self.frame_height), lane_color, 1)

        cv2.line(frame, (0, self.hit_zone_y), (self.frame_width, self.hit_zone_y), (0, 190, 255), 2)
        cv2.putText(frame, "HIT ZONE", (10, self.hit_zone_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 190, 255), 2)

        for note in self.notes:
            color = (90, 220, 90) if note.active else (70, 70, 70)
            radius = 36
            lane_x = self.lane_x[note.lane]
            cv2.circle(frame, (lane_x, int(note.y)), radius, color, -1)
            cv2.circle(frame, (lane_x, int(note.y)), radius, (255, 255, 255), 3)

        cv2.putText(frame, f"Score: {self.score}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, active_color, 2)
        cv2.putText(frame, f"Combo: {self.combo}", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, active_color, 2)
        cv2.putText(frame, f"Health: {self.health}", (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, active_color, 2)

        if hand_info["found"]:
            hand_color = (220, 80, 80) if hand_info["pinch"] else (80, 190, 220)
            cv2.circle(frame, (hand_info["x"], hand_info["y"]), 16, hand_color, -1)
            status_text = "Pinch to hit" if not hand_info["pinch"] else "Hit!"
            cv2.putText(frame, status_text, (hand_info["x"] - 80, hand_info["y"] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

        if current_time - self.last_hit_time < 1.2 and self.last_hit_feedback:
            cv2.putText(frame, self.last_hit_feedback, (self.frame_width - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 180), 3)

        if self.game_over:
            cv2.rectangle(frame, (70, 140), (self.frame_width - 70, self.frame_height - 140), (0, 0, 0), -1)
            cv2.rectangle(frame, (70, 140), (self.frame_width - 70, self.frame_height - 140), (0, 160, 255), 3)
            cv2.putText(frame, "GAME OVER", (self.frame_width // 2 - 170, self.frame_height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 220, 255), 4)
            cv2.putText(frame, f"Final Score: {self.score}", (self.frame_width // 2 - 170, self.frame_height // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 3)
            cv2.putText(frame, "Press R to restart or Q to quit", (self.frame_width // 2 - 260, self.frame_height // 2 + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)

    def reset_game(self) -> None:
        self.notes.clear()
        self.score = 0
        self.combo = 0
        self.health = 6
        self.game_over = False
        self.last_hit_feedback = ""
        self.last_hit_time = 0.0
        self.last_spawn_time = time.time()


if __name__ == '__main__':
    game = RhythmGameWithSound()

    prev_time = time.time()
    while True:
        ret, frame = game.cap.read()
        if not ret:
            break

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = game.landmarker.detect_for_video(mp_image, int(current_time * 1000))
        hand_info = game.extract_hand_info(results, frame.shape)

        if not game.game_over:
            game.spawn_notes(current_time)
            game.update_notes(hand_info, current_time)

        game.draw_game(frame, hand_info, current_time)
        cv2.imshow("Hand Rhythm Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("r") and game.game_over:
            game.reset_game()

    game.cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
