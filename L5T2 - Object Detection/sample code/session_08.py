"""
Hand Rhythm Game: A hand gesture-based rhythm game using MediaPipe hand detection.
Player pinches fingers to hit falling notes that must be timed to reach the hit zone.
Uses webcam input and hand landmarks for gesture recognition.
"""

import cv2
import time
import random
import math
from dataclasses import dataclass
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions

@dataclass
class Note:
    """Represents a falling note in the rhythm game.
    
    Attributes:
        lane: Which of the 3 lanes (0-2) the note spawns in
        y: Vertical position on screen (0 = top, increases downward)
        speed: Pixels per second the note falls
        active: Whether the note is still in play
    """
    lane: int
    y: float
    speed: float = 30.0
    active: bool = True

    def update(self, dt: float) -> None:
        """Update note position. Deactivate if it falls off screen."""
        self.y += self.speed * dt
        if self.y > 540:
            self.active = False


class RhythmGame:
    """Main game controller managing gameplay, hand detection, and rendering."""
    
    def __init__(self) -> None:
        """Initialize game state, webcam, and hand detection model."""
        # Set up webcam capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam. Check camera permissions and device index.")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        # Initialize MediaPipe HandLandmarker for real-time hand detection
        model_path = "hand_landmarker.task"
        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,  # Only track one hand
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Set up 3-lane gameplay layout
        self.lane_count = 3
        self.lane_x = [int((i + 0.5) * self.frame_width / self.lane_count) for i in range(self.lane_count)]
        
        # Define hit zone: where notes must be to register as hits
        self.hit_zone_y = int(self.frame_height * 0.82)  # Near bottom of screen
        self.hit_window = 80  # ±80 pixels around hit zone for timing tolerance
        
        # Note spawning parameters
        self.spawn_interval = 0.8  # Spawn new note every 0.8 seconds
        self.last_spawn_time = time.time()

        # Game state tracking
        self.notes = []  # Active notes currently falling
        self.score = 0
        self.combo = 0  # Consecutive successful hits
        self.health = 6  # Lives (game over when reaches 0)
        self.moves = 0  # Total notes hit
        self.game_over = False
        
        self.last_hit_feedback = ""  # Display feedback (Perfect/Great/Good)
        self.last_hit_time = 0.0  # When last hit feedback was shown
        self.start_time = time.time()

    def run(self) -> None:
        """Main game loop: capture frames, detect hands, update game state, and render."""
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Calculate delta time for physics updates
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            # Prepare frame for hand detection (mirror and convert to RGB)
            frame = cv2.flip(frame, 1)  # Mirror horizontally for intuitive hand control
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run hand detection using MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect_for_video(mp_image, int(current_time * 1000))
            hand_info = self.extract_hand_info(results, frame.shape)

            # Update game state only if not game over
            if not self.game_over:
                self.spawn_notes(current_time)  # Randomly spawn new falling notes
                self.update_notes(hand_info, current_time)  # Check hits and update note positions

            # Render all game elements on frame
            self.draw_game(frame, hand_info, current_time)
            cv2.imshow("Hand Rhythm Game", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC to quit
                break
            if key == ord("r") and self.game_over:  # R to restart when game over
                self.reset_game()

        self.cap.release()
        cv2.destroyAllWindows()

    def extract_hand_info(self, results, shape) -> dict:
        """Extract hand position and pinch gesture from detection results.
        
        Returns dict with: found (bool), x, y (pixel coords), pinch (bool), landmarks
        """
        hand_info = {"found": False, "x": 0, "y": 0, "pinch": False}
        if not results.hand_landmarks:
            return hand_info  # No hand detected

        # HandLandmarker returns 21 landmarks per detected hand
        hand_lms = results.hand_landmarks[0]  # Get first (only) hand
        width, height = shape[1], shape[0]

        # Extract key finger positions for pinch detection
        # Landmark indices: 8 = INDEX_FINGER_TIP, 4 = THUMB_TIP
        index_tip = hand_lms[8]
        thumb_tip = hand_lms[4]
        
        # Convert normalized coordinates (0-1) to pixel coordinates
        px = int(index_tip.x * width)
        py = int(index_tip.y * height)

        # Calculate distance between thumb and index finger to detect pinch
        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        pinch = pinch_dist < 0.055  # Threshold for pinch gesture

        hand_info.update({"found": True, "x": px, "y": py, "pinch": pinch})
        hand_info["landmarks"] = hand_lms
        return hand_info

    def spawn_notes(self, current_time: float) -> None:
        """Randomly spawn a new note if spawn interval has elapsed."""
        if current_time - self.last_spawn_time < self.spawn_interval:
            return  # Not time to spawn yet

        # Pick random lane and create note above screen
        lane = random.randint(0, self.lane_count - 1)
        self.notes.append(Note(lane=lane, y=-50.0, speed=40.0))
        self.last_spawn_time = current_time

    def update_notes(self, hand_info: dict, current_time: float) -> None:
        """Update all falling notes: check for hits and clean up missed notes."""
        for note in self.notes:
            if not note.active:
                continue
                
            # Update note position
            note.update(current_time - self.last_spawn_time)

            # Check if note passed the hit zone without being hit (miss)
            if note.y > self.hit_zone_y + self.hit_window:
                note.active = False
                self.combo = 0  # Break combo on miss
                self.health -= 1  # Lose a life
                self.last_hit_feedback = "Miss"
                self.last_hit_time = current_time
                continue

            # Check if hand is pinching and in the right lane at the right time
            if hand_info["found"] and hand_info["pinch"] and self.is_note_in_lane(note, hand_info):
                distance = abs(note.y - self.hit_zone_y)  # How far from perfect timing
                if distance < self.hit_window:  # Within hit window
                    note.active = False
                    
                    # Score based on timing accuracy (Perfect > Great > Good)
                    points = 120 if distance < 30 else 90 if distance < 55 else 60
                    self.score += points + self.combo * 5  # Combo multiplier
                    self.combo += 1
                    self.moves += 1
                    
                    # Display hit feedback
                    self.last_hit_feedback = "Perfect" if distance < 30 else "Great" if distance < 55 else "Good"
                    self.last_hit_time = current_time

        # Remove all inactive notes
        self.notes = [note for note in self.notes if note.active]
        
        # Check game over condition
        if self.health <= 0:
            self.game_over = True

    def is_note_in_lane(self, note: Note, hand_info: dict) -> bool:
        """Check if hand is positioned in the same lane as the note."""
        lane_center = self.lane_x[note.lane]
        lane_width = self.frame_width // (self.lane_count * 2)  # Half the lane width
        return abs(hand_info["x"] - lane_center) < lane_width

    def draw_game(self, frame, hand_info: dict, current_time: float) -> None:
        """Render all visual elements: lanes, notes, UI, and hand cursor."""
        lane_color = (180, 180, 180)
        active_color = (255, 255, 255)

        # Draw lane dividers
        for x in self.lane_x:
            cv2.line(frame, (x, 0), (x, self.frame_height), lane_color, 1)

        # Draw hit zone line where notes must be hit
        cv2.line(frame, (0, self.hit_zone_y), (self.frame_width, self.hit_zone_y), (0, 190, 255), 2)
        cv2.putText(frame, "HIT ZONE", (10, self.hit_zone_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 190, 255), 2)

        # Draw all active falling notes as circles
        for note in self.notes:
            color = (90, 220, 90) if note.active else (70, 70, 70)  # Green if active
            radius = 36
            lane_x = self.lane_x[note.lane]
            cv2.circle(frame, (lane_x, int(note.y)), radius, color, -1)  # Filled circle
            cv2.circle(frame, (lane_x, int(note.y)), radius, (255, 255, 255), 3)  # White outline

        # Draw game stats in top-left corner
        cv2.putText(frame, f"Score: {self.score}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, active_color, 2)
        cv2.putText(frame, f"Combo: {self.combo}", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, active_color, 2)
        cv2.putText(frame, f"Health: {self.health}", (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, active_color, 2)
        cv2.putText(frame, f"Moves: {self.moves}", (16, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.8, active_color, 2)

        # Draw hand cursor indicator
        if hand_info["found"]:
            # Red if pinching, blue if not
            hand_color = (220, 80, 80) if hand_info["pinch"] else (80, 190, 220)
            cv2.circle(frame, (hand_info["x"], hand_info["y"]), 16, hand_color, -1)
            
            # Show pinch status
            status_text = "Pinch to hit" if not hand_info["pinch"] else "Hit!"
            cv2.putText(frame, status_text, (hand_info["x"] - 80, hand_info["y"] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

        # Display hit feedback message (Perfect/Great/Good) for 1.2 seconds
        if current_time - self.last_hit_time < 1.2 and self.last_hit_feedback:
            cv2.putText(frame, self.last_hit_feedback, (self.frame_width - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 180), 3)

        # Draw on-screen instructions
        self.draw_instructions(frame)

        # Draw game over screen
        if self.game_over:
            # Semi-transparent black overlay
            cv2.rectangle(frame, (70, 140), (self.frame_width - 70, self.frame_height - 140), (0, 0, 0), -1)
            cv2.rectangle(frame, (70, 140), (self.frame_width - 70, self.frame_height - 140), (0, 160, 255), 3)
            
            # Game over text and stats
            cv2.putText(frame, "GAME OVER", (self.frame_width // 2 - 170, self.frame_height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 220, 255), 4)
            cv2.putText(frame, f"Final Score: {self.score}", (self.frame_width // 2 - 170, self.frame_height // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 3)
            cv2.putText(frame, "Press R to restart or Q to quit", (self.frame_width // 2 - 260, self.frame_height // 2 + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)

    def draw_instructions(self, frame) -> None:
        """Draw on-screen help text."""
        cv2.putText(frame, "Hand Rhythm Game", (self.frame_width - 330, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 100), 2)
        cv2.putText(frame, "Move your hand to a lane and pinch to hit notes.", (16, self.frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(frame, "Q = Quit | R = Restart", (16, self.frame_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    def reset_game(self) -> None:
        """Reset all game state for a new game."""
        self.notes.clear()
        self.score = 0
        self.combo = 0
        self.health = 6
        self.moves = 0
        self.game_over = False
        self.last_hit_feedback = ""
        self.last_hit_time = 0.0
        self.start_time = time.time()
        self.last_spawn_time = time.time()


if __name__ == "__main__":
    # Initialize and start the game
    game = RhythmGame()
    game.run()
