"""
Combined UGOT + Webcam Demo with Gesture & Object Detection

Display both webcam (gesture HUD) and UGOT camera (YOLO HUD) side-by-side.
Controls:
  'g' = gesture control mode (webcam HUD, gesture -> robot movement)
  'o' = object detection mode (UGOT HUD, YOLO -> approach target)
  'q' = quit
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from ugot import ugot
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================
UGOT_IP = "192.168.1.230"
YOLO_MODEL = "../IMDA/best_coffee.pt"
TARGET_HEIGHT = 480
WEBCAM_INDEX = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize UGOT robot
got = ugot.UGOT()
got.initialize(UGOT_IP)
got.open_camera()

# Load YOLO model
trained = YOLO(YOLO_MODEL)

# ============================================================================
# GESTURE RECOGNITION HELPERS
# ============================================================================
def is_finger_up(tip_idx, pip_idx, lm):
    return lm[tip_idx].y < lm[pip_idx].y

def count_fingers(lm):
    """Count extended fingers (index, middle, ring, pinky) on a hand."""
    fingers = [is_finger_up(8,6,lm), is_finger_up(12,10,lm), is_finger_up(16,14,lm), is_finger_up(20,18,lm)]
    return sum(1 for f in fingers if f)

def classify_right_hand(lm):
    """Classify right hand gesture to direction command.
    
    Returns: "Forward", "Backward", "Left", "Right", or "None"
    """
    thumb_up = lm[4].x < lm[3].x
    four_bits = [1 if is_finger_up(8,6,lm) else 0,
                 1 if is_finger_up(12,10,lm) else 0,
                 1 if is_finger_up(16,14,lm) else 0,
                 1 if is_finger_up(20,18,lm) else 0]
    if four_bits == [0,0,0,0]:
        return "Backward" if thumb_up else "Left"
    if four_bits == [1,0,0,0]:
        return "Right"
    if four_bits == [1,1,1,1]:
        return "Forward"
    return "None"

def speed_from_fingers(fingers):
    """Map number of extended fingers to speed (0-50)."""
    table = {0:0, 1:10, 2:20, 3:30, 4:40, 5:50}
    return table.get(fingers, 0)

# ============================================================================
# VISION HELPERS
# ============================================================================

def draw_detections(frame, results):
    """Draw YOLO bounding boxes and labels on frame."""
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# ============================================================================
# ROBOT CONTROL
# ============================================================================

def control_robot(direction, speed):
    """Send movement command to UGOT based on direction and speed."""
    if speed == 0 or direction == "None":
        got.balance_stop_balancing()
        return
    if direction == "Forward":
        got.balance_move_speed(0, speed)
    elif direction == "Backward":
        got.balance_move_speed(1, speed)
    elif direction == "Left":
        got.balance_turn_speed(2, speed * 2)
    elif direction == "Right":
        got.balance_turn_speed(3, speed * 2)

# ============================================================================
# MODE-SPECIFIC PROCESSING FUNCTIONS
# ============================================================================
def process_gesture_frame(frame, hands):
    """Process webcam frame for gesture recognition.
    
    Args:
        frame: BGR image from webcam (already flipped)
        hands: MediaPipe Hands detector
        
    Returns:
        (annotated_frame, direction, speed)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    direction = "None"
    speed = 0
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            if handedness.classification[0].label == 'Left':
                speed = speed_from_fingers(count_fingers(lm))
            else:
                direction = classify_right_hand(lm)

    cv2.putText(frame, f"Speed: {speed}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame, direction, speed


def detect_and_approach_object(frame, trained):
    """Process UGOT frame for object detection and approach.
    
    Runs YOLO detection, finds best target, and controls robot movement.
    
    Args:
        frame: BGR image from UGOT
        trained: YOLO model
        
    Returns:
        annotated_frame (with bounding boxes and HUD)
    """
    try:
        results = trained(frame, verbose=False)
    except Exception:
        results = []
    
    output = draw_detections(frame, results)

    # Find best detection
    max_conf = 0.0
    x = y = w = h = 0.5
    found = False
    
    for r in results:
        detected = r.boxes.cls.tolist()
        confidences = r.boxes.conf.tolist()
        xywhn = r.boxes.xywhn.tolist()
        for idx, cls_id in enumerate(detected):
            conf = confidences[idx]
            if conf > max_conf:
                max_conf = conf
                x, y, w, h = xywhn[idx]
                found = True

    # Draw HUD
    if found:
        area = w * h
        cv2.putText(output, f"Centre: ({x:.3f}, {y:.3f})", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, f"Area: {area:.3f}", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, f"Confidence: {max_conf:.3f}", (30, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Robot control via approach logic
    approach_object(x, y, w, h, found)
    
    return output


def approach_object(x, y, w, h, found):
    """Control robot movement to approach detected object.
    
    Customize this function to change approach behavior:
    - x, y: normalized center coordinates (0..1)
    - w, h: normalized box dimensions
    - found: whether an object was detected
    
    Edit this function to change how the robot approaches objects.
    """
    if not found:
        # No object: scan by rotating
        got.balance_turn_speed(3, 10)
        return
    
    area = w * h
    
    # Horizontal centering: turn to face object
    if x > 0.6:
        got.balance_move_turn(0, 5, 3, 10)  # turn right
    elif x < 0.4:
        got.balance_move_turn(0, 5, 2, 10)  # turn left
    else:
        # Centered: move forward/backward based on distance
        if area < 0.03:
            got.balance_move_speed(0, 15)  # move forward (far)
        elif area < 0.06:
            got.balance_move_speed(0, 6)   # move forward (medium)
        elif area > 0.2:
            got.balance_move_speed(1, 6)   # move backward (too close)
        else:
            got.balance_stop_balancing()   # target distance reached


# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    """Main loop: display webcam and UGOT side-by-side, switch modes with keys."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    got.balance_start_balancing()
    time.sleep(0.5)

    mode = 'gesture'  # start in gesture control mode

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            # Read frames from both cameras
            ok, webcam = cap.read()
            if not ok:
                webcam = np.zeros((TARGET_HEIGHT, TARGET_HEIGHT, 3), dtype=np.uint8)
            webcam = cv2.flip(webcam, 1)

            data = got.read_camera_data()
            if data is not None:
                nparr = np.frombuffer(data, np.uint8)
                ugot_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                ugot_frame = np.zeros((TARGET_HEIGHT, TARGET_HEIGHT, 3), dtype=np.uint8)

            # Process frames based on mode
            if mode == 'gesture':
                webcam_disp, direction, speed = process_gesture_frame(webcam, hands)
                control_robot(direction, speed)
                ugot_disp = ugot_frame
            else:
                ugot_disp = detect_and_approach_object(ugot_frame, trained)
                webcam_disp = webcam

            # Compose and display side-by-side
            th = min(TARGET_HEIGHT, ugot_disp.shape[0], webcam_disp.shape[0])
            ua = ugot_disp.shape[1] / ugot_disp.shape[0]
            wa = webcam_disp.shape[1] / webcam_disp.shape[0]
            ug = cv2.resize(ugot_disp, (int(th * ua), th))
            wb = cv2.resize(webcam_disp, (int(th * wa), th))
            combined = np.hstack((ug, wb))
            
            # Overlay labels
            cv2.putText(combined, f"MODE: {mode.upper()}", (10, combined.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(combined, "UGOT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "WEBCAM", (ug.shape[1]+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Dual View", combined)

            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                mode = 'object'
                got.balance_stop_balancing()
                print("Switched to object detection mode (press 'g' for gesture)")
            elif key == ord('g'):
                mode = 'gesture'
                got.balance_stop_balancing()
                print("Switched to gesture control mode (press 'o' for object)")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        got.balance_stop_balancing()


if __name__ == '__main__':
    main()