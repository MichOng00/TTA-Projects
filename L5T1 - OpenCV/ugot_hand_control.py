"""
Hand gesture-based UGOT robot control using MediaPipe Hand Skeleton
Left hand: Speed control (hand height)
Right hand: Direction control (pointing gestures - forward/backward/turn left/turn right)
"""
import cv2
import mediapipe as mp
import numpy as np
from ugot import ugot
import time

# Initialize UGOT robot
got = ugot.UGOT()
got.initialize("192.168.1.106")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Hand gesture state
move_speed = 30  # Default movement speed


def lmk_xy(hand_landmarks, idx, w, h):
    """Extract x, y coordinates from hand landmark at given index."""
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def euclid(a, b):
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def get_hand_speed(hand_landmarks, w, h):
    """
    Calculate speed from left hand height.
    Hand higher on screen = faster movement.
    Range: 10 cm/s at bottom to 100 cm/s at top
    """
    wrist = lmk_xy(hand_landmarks, 0, w, h)  # Wrist is landmark 0
    
    # Normalize height (0 = top of screen, 1 = bottom)
    # Lower y = higher on screen
    normalized_height = 1.0 - (wrist[1] / h)
    
    # Map to speed range [10, 100]
    speed = int(10 + normalized_height * 90)
    speed = max(10, min(100, speed))  # Clamp to valid range
    
    return speed, wrist


def count_fingers_up(hand_landmarks, w, h):
    """
    Count how many fingers are up (extended).
    Returns: number of fingers (0-5)
    """
    # Landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    # Tip indices: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
    # PIP indices: 3=thumb, 6=index, 10=middle, 14=ring, 18=pinky (middle joints)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    fingers_up = 0
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = lmk_xy(hand_landmarks, tip_idx, w, h)
        pip = lmk_xy(hand_landmarks, pip_idx, w, h)
        if tip[1] < pip[1]:
            fingers_up += 1
    return fingers_up






# New execute_command using fingers_up and speed
def execute_command(fingers_up, speed):
    if fingers_up == 1:
        got.mecanum_move_speed(0, speed) # forward
        print(f"→ Forward ({speed})")
    elif fingers_up == 2:
        got.mecanum_move_speed(1, speed) # backward
        print(f"← Backward ({speed})")
    elif fingers_up == 3:
        got.mecanum_turn_speed(2, speed) # left
        print(f"⟲ Turn Left")
    elif fingers_up == 4:
        got.mecanum_turn_speed(3, speed) # right
        print(f"⟳ Turn Right")
    else:
        got.mecanum_stop()
        print("Stopped")
 


# Main loop
try:
    print("Starting UGOT hand control...")
    print("ESC to exit")
    
    while True:
        # Get webcam frame
        ret, frame = cap.read()
        if not ret:
            break
        # Flip frame horizontally to correct left/right
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        # Detect hands
        results = hands.process(frame_rgb)
        
        # Assign hands by handedness
        left_speed = move_speed
        left_info = ""
        gesture_info = ""
        right_fingers = None
        left_fingers = None
        right_wrist = None
        left_wrist = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness_info.classification[0].label
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(255, 0, 0), thickness=2
                    )
                )
                if hand_label == "Right":
                    right_fingers = count_fingers_up(hand_landmarks, w, h)
                    right_wrist = lmk_xy(hand_landmarks, 0, w, h)
                elif hand_label == "Left":
                    left_fingers = count_fingers_up(hand_landmarks, w, h)
                    left_wrist = lmk_xy(hand_landmarks, 0, w, h)
        # Speed from left hand, command from right hand
        if left_wrist is not None:
            left_speed, _ = get_hand_speed(hand_landmarks, w, h) if left_fingers is not None else (move_speed, None)
            left_info = f"LEFT HAND - Speed: {left_speed}cm/s, Fingers: {left_fingers if left_fingers is not None else 0}"
            cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 15, (0, 255, 255), -1)
            cv2.putText(
                frame, f"{left_speed}", (int(left_wrist[0]) - 20, int(left_wrist[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        if right_fingers is not None:
            gesture_info = f"RIGHT HAND - Fingers: {right_fingers}"
            execute_command(right_fingers, left_speed)
        # ...existing code for HUD and display...
        
        # Display HUD
        y_offset = 30
        cv2.putText(
            frame, "Hand Control: L=Speed, R=Direction",
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        y_offset += 30
        if left_info:
            cv2.putText(
                frame, left_info, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
        
        y_offset += 30
        if gesture_info:
            cv2.putText(
                frame, gesture_info, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        # Display frame
        cv2.imshow("Hand Control UGOT", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC
            break

finally:
    print("Shutting down...")
    got.mecanum_stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Control ended.")