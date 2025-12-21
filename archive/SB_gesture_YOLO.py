"""
NYTC 2026 - Gesture Control + Object Detection for UGOT Robot (Self-Balancing Vehicle)

This module combines two functionalities:
1. **Gesture Recognition**: Uses MediaPipe to detect hand gestures from a webcam
   - RIGHT hand controls DIRECTION (forward/backward/left/right)
   - LEFT hand controls SPEED (0-4 fingers = 0-40 speed)

2. **Object Detection**: Uses YOLOv11 custom model to approach objects with UGOT camera
   - Finds the highest confidence detection
   - Centers and approaches the object to a target distance

This code assumes you have already trained and saved a model on your own dataset.
"""

import cv2
import mediapipe as mp
from ugot import ugot
import time
import numpy as np
from ultralytics import YOLO

# ============================================================================
# MEDIAPIPE HAND GESTURE SETUP
# ============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand landmark indices (MediaPipe convention):
# Fingers: 8=index_tip, 6=index_pip, 12=middle_tip, 10=middle_pip, etc.
# Thumb: 4=thumb_tip, 3=thumb_ip


# ============================================================================
# HELPER FUNCTIONS: HAND GESTURE RECOGNITION
# ============================================================================

def count_fingers(lm):
    """
    Count how many fingers are extended on a hand.
    
    Only counts INDEX, MIDDLE, RING, PINKY (thumb is ignored for reliability).
    A finger is considered "up" if its tip is higher on screen than its PIP joint.
    
    Args:
        lm: MediaPipe hand_landmarks.landmark list (21 joints)
        
    Returns:
        int: Number of fingers up (0-4)
    """
    def is_finger_up(tip_idx, pip_idx):
        """
        Finger is up if tip.y < pip.y (smaller y = higher on screen in OpenCV).
        """
        return lm[tip_idx].y < lm[pip_idx].y

    index_up = is_finger_up(8, 6)
    middle_up = is_finger_up(12, 10)
    ring_up = is_finger_up(16, 14)
    pinky_up = is_finger_up(20, 18)

    fingers = [index_up, middle_up, ring_up, pinky_up]
    return sum(1 for f in fingers if f)


def classify_right_hand(lm):
    """
    Classify RIGHT hand gesture to determine robot direction.
    
    Gesture Mapping:
        - Fist (all fingers down, thumb not extended) → LEFT
        - 1 finger up (index only) → RIGHT
        - All 4 fingers up (index-pinky extended) → FORWARD
        - Thumb only (fist + thumb extended) → BACKWARD
        - Unrecognized → NONE
    
    Args:
        lm: MediaPipe hand_landmarks.landmark list
        
    Returns:
        str: Direction command ("Left", "Right", "Forward", "Backward", or "None")
    """
    def is_finger_up(tip_idx, pip_idx):
        """Helper: check if finger is extended."""
        return lm[tip_idx].y < lm[pip_idx].y

    # Thumb extended if thumb_tip.x < thumb_ip.x (for right hand)
    thumb_up = lm[4].x < lm[3].x

    # Check index-pinky fingers
    index_up = is_finger_up(8, 6)
    middle_up = is_finger_up(12, 10)
    ring_up = is_finger_up(16, 14)
    pinky_up = is_finger_up(20, 18)

    # Convert to binary pattern [0/1, 0/1, 0/1, 0/1]
    four_fingers = [index_up, middle_up, ring_up, pinky_up]
    four_bits = [1 if f else 0 for f in four_fingers]

    # === GESTURE PATTERNS ===
    
    # Fist: no index-pinky fingers up
    if four_bits == [0, 0, 0, 0]:
        return "Backward" if thumb_up else "Left"

    # Index only up → RIGHT
    if four_bits == [1, 0, 0, 0]:
        return "Right"

    # All 4 fingers (index-pinky) up → FORWARD
    if four_bits == [1, 1, 1, 1]:
        return "Forward"

    # Unrecognized pattern
    return "None"


def speed_from_fingers(fingers):
    """
    Map number of extended fingers to robot speed.
    
    Mapping:
        0 fingers (fist) → 0 (stop)
        1 finger → 10
        2 fingers → 20
        3 fingers → 30
        4 fingers → 40
    
    Args:
        fingers (int): Number of extended fingers (0-4)
        
    Returns:
        int: Speed value (0-50)
    """
    speed_table = {
        0: 0,
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,  # Safety fallback (should not occur)
    }
    return speed_table.get(fingers, 0)


class GestureYOLOProcessor:
    """
    Processor that provides two callables:
      - process_webcam(frame): annotate webcam frame with gesture HUD and send robot commands
      - process_ugot(frame): annotate UGOT frame with YOLO detections and optionally send robot commands

    This class initializes MediaPipe and the YOLO model once and is safe to call per-frame.
    """

    def __init__(self, ugot_ip="192.168.1.230", yolo_path="../IMDA/best_coffee.pt"):
        self.got = ugot.UGOT()
        try:
            self.got.initialize(ugot_ip)
        except Exception:
            # if initialization fails, we still allow processing/display (robot commands will no-op)
            pass

        # Load YOLO model
        self.trained = YOLO(yolo_path)

        # MediaPipe Hands instance (kept alive)
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def control_robot(self, direction, speed):
        """Send movement commands to the UGOT robot using this instance's `got`."""
        if not hasattr(self, 'got') or self.got is None:
            return

        if speed == 0 or direction == "None":
            self.got.balance_stop_balancing()
            return

        if direction == "Forward":
            self.got.balance_move_speed(0, speed)
        elif direction == "Backward":
            self.got.balance_move_speed(1, speed)
        elif direction == "Left":
            self.got.balance_turn_speed(2, speed * 2)
        elif direction == "Right":
            self.got.balance_turn_speed(3, speed * 2)

    def process_webcam(self, frame):
        """Annotate a webcam BGR frame with gesture HUD and send robot commands."""
        # Mirror image for intuitive UX
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        # Delegate to gesture_logic hook to decide direction & speed from MediaPipe results.
        # Novice developers: edit `gesture_logic` below to change gesture->command mapping.
        direction, speed = self.gesture_logic(results)

        # Execute robot movement (safe wrapper)
        self.control_robot(direction, speed)

        # draw landmarks and HUD (kept for display)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Speed: {speed}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def gesture_logic(self, mp_results):
        """Hook: decide direction and speed from MediaPipe `Hands` results.

        Args:
            mp_results: the object returned by `self.hands.process(rgb)` (MediaPipe results)

        Returns:
            (direction: str, speed: int)

        Default behaviour:
          - LEFT hand controls `speed` via `speed_from_fingers`.
          - RIGHT hand controls `direction` via `classify_right_hand`.

        To customize, change the body of this method to inspect landmarks and
        return a different `(direction, speed)` tuple.
        """
        direction = "None"
        speed = 0

        if mp_results.multi_hand_landmarks and mp_results.multi_handedness:
            for hand_landmarks, handedness in zip(
                mp_results.multi_hand_landmarks, mp_results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                lm = hand_landmarks.landmark

                if hand_label == "Left":
                    left_fingers = count_fingers(lm)
                    speed = speed_from_fingers(left_fingers)

                elif hand_label == "Right":
                    direction = classify_right_hand(lm)

        return direction, speed

    def process_ugot(self, frame):
        """Run YOLO on UGOT frame, draw detections and optionally control robot.

        Expects BGR image as input and returns annotated BGR image.
        """
        img = frame

        # Run YOLO detection
        try:
            results = self.trained(img, verbose=False)
        except Exception:
            return frame

        # Draw bounding boxes and find highest-confidence detection
        max_conf = 0.0
        max_x = 0.5
        max_y = 0.5
        max_w = 0.0
        max_h = 0.0
        found = False

        for r in results:
            boxes = r.boxes

            # draw all boxes for visualization
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # use normalized xywh to pick best detection
            xywhn = r.boxes.xywhn.tolist()
            detected = r.boxes.cls.tolist()
            confidences = r.boxes.conf.tolist()
            for idx, cls_id in enumerate(detected):
                conf = confidences[idx]
                cx, cy, bw, bh = xywhn[idx]
                if conf > max_conf:
                    max_conf = conf
                    max_x = cx
                    max_y = cy
                    max_w = bw
                    max_h = bh
                    found = True

        # Delegate approach decision to `approach_logic` hook
        # `approach_logic` returns a dict describing desired action.
        action = self.approach_logic(max_x, max_y, max_w, max_h, max_conf, found)

        # Draw HUD values
        if found:
            area = max_w * max_h
            cv2.putText(img, f"Centre: ({max_x:.3f}, {max_y:.3f})", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Area: {area:.3f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Confidence: {max_conf:.3f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Map action to robot API (kept same behavior by default)
        if action is not None:
            act = action.get('action')
            speed = action.get('speed', 10)

            if act == 'turn_right':
                self.got.balance_move_turn(0, 5, 3, speed)
            elif act == 'turn_left':
                self.got.balance_move_turn(0, 5, 2, speed)
            elif act == 'move_forward':
                self.got.balance_move_speed(0, speed)
            elif act == 'move_backward':
                self.got.balance_move_speed(1, speed)
            elif act == 'stop':
                self.got.balance_stop_balancing()
            elif act == 'scan':
                self.got.balance_turn_speed(3, speed)

        return img

    def approach_logic(self, cx, cy, w, h, confidence, found):
        """Hook: decide approach behaviour from best detection.

        Args:
            cx, cy: normalized center coordinates (0..1)
            w, h: normalized box width/height
            confidence: detection confidence (0..1)
            found: bool whether any detection was found

        Returns:
            dict or None, e.g. { 'action': 'turn_right', 'speed': 10 }

        Default behaviour (editable by a novice):
          - If no detection -> return {action: 'scan', speed: 10}
          - If detection off-center -> return 'turn_left' or 'turn_right'
          - If centered -> return 'move_forward' with speed depending on area
          - If too close -> 'stop' or 'move_backward'
        """
        if not found:
            return {'action': 'scan', 'speed': 10}

        area = w * h
        # center thresholds and areas
        if cx > 0.6:
            return {'action': 'turn_right', 'speed': 10}
        elif cx < 0.4:
            return {'action': 'turn_left', 'speed': 10}
        else:
            if area < 0.03:
                return {'action': 'move_forward', 'speed': 15}
            elif area < 0.06:
                return {'action': 'move_forward', 'speed': 6}
            elif area > 0.2:
                return {'action': 'move_backward', 'speed': 6}
            else:
                return {'action': 'stop', 'speed': 0}
