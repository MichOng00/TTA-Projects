"""
Hand Rhythm Game - Draw Only Version
Visual rendering without game logic. Useful for testing UI and hand detection.
Shows lanes, hit zone, and hand cursor without spawning notes or tracking score.
"""

import cv2
import time
import math
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions


class RhythmGameDrawOnly:
    """Game renderer with hand detection - no gameplay logic."""
    
    def __init__(self) -> None:
        """Initialize webcam and hand detection model."""
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

    def run(self) -> None:
        """Main loop: capture frames, detect hands, and render without game logic."""
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Calculate delta time for consistency
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

            # Render all visual elements on frame
            self.draw_game(frame, hand_info)
            cv2.imshow("Hand Rhythm Game - Draw Only", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC to quit
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def extract_hand_info(self, results, shape) -> dict:
        """Extract hand position and pinch gesture from detection results.
        
        Returns dict with: found (bool), x, y (pixel coords), pinch (bool)
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
        return hand_info

    def draw_game(self, frame, hand_info: dict) -> None:
        """Render all visual elements: lanes, hit zone, and hand cursor."""
        lane_color = (180, 180, 180)
        active_color = (255, 255, 255)

        # Draw lane dividers
        for x in self.lane_x:
            cv2.line(frame, (x, 0), (x, self.frame_height), lane_color, 1)

        # Draw hit zone line where notes must be hit
        cv2.line(frame, (0, self.hit_zone_y), (self.frame_width, self.hit_zone_y), (0, 190, 255), 2)
        cv2.putText(frame, "HIT ZONE", (10, self.hit_zone_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 190, 255), 2)

        # Draw game title
        cv2.putText(frame, "Hand Rhythm Game - Draw Only Mode", (self.frame_width - 450, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 100), 2)
        
        # Draw instructions
        cv2.putText(frame, "Move your hand to a lane. Pinch to test hit detection.", (16, self.frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(frame, "Q = Quit", (16, self.frame_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        # Draw hand cursor indicator
        if hand_info["found"]:
            # Red if pinching, blue if not
            hand_color = (220, 80, 80) if hand_info["pinch"] else (80, 190, 220)
            cv2.circle(frame, (hand_info["x"], hand_info["y"]), 16, hand_color, -1)
            
            # Show pinch status
            status_text = "Pinch to hit" if not hand_info["pinch"] else "PINCHING!"
            cv2.putText(frame, status_text, (hand_info["x"] - 80, hand_info["y"] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            
            # Show hand position coordinates
            cv2.putText(frame, f"X: {hand_info['x']}, Y: {hand_info['y']}", (hand_info["x"] - 80, hand_info["y"] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)


if __name__ == "__main__":
    # Initialize and start the draw-only renderer
    game = RhythmGameDrawOnly()
    game.run()
