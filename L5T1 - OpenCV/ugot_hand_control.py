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
    Returns: number of fingers (0-5), confidence (0.0-1.0)
    
    Finger mapping:
    - 1 finger up = forward
    - 2 fingers up = backward
    - 3 fingers up = left
    - 4 fingers up = right
    """
    # Landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    # Tip indices: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
    # PIP indices: 3=thumb, 6=index, 10=middle, 14=ring, 18=pinky (middle joints)
    
    finger_tips = [8, 12, 16, 20]          # Thumb, Index, Middle, Ring, Pinky
    finger_pips = [6, 10, 14, 18]          # PIP joints for each finger
    
    fingers_up = 0
    
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = lmk_xy(hand_landmarks, tip_idx, w, h)
        pip = lmk_xy(hand_landmarks, pip_idx, w, h)
        
        # Finger is up if tip is above (lower y value than) the PIP joint
        if tip[1] < pip[1]:
            fingers_up += 1
    
    confidence = 0.8  # Relatively high confidence for finger counting
    
    # Map finger count to gesture
    gesture = "idle"
    if fingers_up == 1:
        gesture = "forward"
    elif fingers_up == 2:
        gesture = "backward"
    elif fingers_up == 3:
        gesture = "left"
    elif fingers_up == 4:
        gesture = "right"
    
    return gesture, confidence


def detect_right_hand_gesture(hand_landmarks, w, h, handedness):
    """
    Detect hand gesture from right hand landmarks based on fingers up.
    Returns: gesture_type, confidence
    """
    return count_fingers_up(hand_landmarks, w, h)


def execute_command(gesture, speed, handedness):
    """Execute robot movement based on detected gesture."""    
    # Execute movement command
    if gesture == "forward":
        got.mecanum_move_speed(0, speed)
        print(f"→ Forward ({speed})")
    elif gesture == "backward":
        got.mecanum_move_speed(1, speed)
        print(f"← Backward ({speed})")
    elif gesture == "left":
        got.mecanum_turn_speed(2, speed)
        print(f"⟲ Turn Left")
    elif gesture == "right":
        got.mecanum_turn_speed(3, speed)
        print(f"⟳ Turn Right")
    else:
        got.mecanum_stop()
 


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
        
        left_hand = None
        right_hand = None
        left_handedness = None
        right_handedness = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness_info.classification[0].label
                
                if hand_label == "Left":
                    left_hand = hand_landmarks
                    left_handedness = hand_label
                else:
                    right_hand = hand_landmarks
                    right_handedness = hand_label
                
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
        
        # Process left hand for speed control
        left_speed = move_speed
        left_info = ""
        if left_hand:
            left_speed, wrist_pos = get_hand_speed(left_hand, w, h)
            left_info = f"LEFT HAND - Speed: {left_speed}cm/s"
            # Draw speed indicator
            cv2.circle(frame, (int(wrist_pos[0]), int(wrist_pos[1])), 15, (0, 255, 255), -1)
            cv2.putText(
                frame, f"{left_speed}", (int(wrist_pos[0]) - 20, int(wrist_pos[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        
        # Process right hand for direction control
        if right_hand:
            gesture, gesture_confidence = detect_right_hand_gesture(right_hand, w, h, right_handedness)
            gesture_info = f"RIGHT HAND - Gesture: {gesture.upper()} ({gesture_confidence:.2f})"
            
            # Execute movement if confidence is high enough
            if gesture_confidence > 0.2:
                execute_command(gesture, left_speed, right_handedness)
        else:
            gesture = "idle"
            gesture_confidence = 0.0
            gesture_info = ""
        
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