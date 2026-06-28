"""
Webcam Gesture Control - Hand Gesture Classification (Fist, 1-Finger, Open, Thumb)
Similar to NYTC_2026_SB_gesture_YOLO but uses webcam instead of UGOT camera.
RIGHT hand controls direction using gesture classification.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

show_labels = True
last_action = None

def forward():
    print("go forward")

def backward():
    print("go back")

def turn_left():
    print("go left")

def turn_right():
    print("go right")

def stop():
    print("stop")

def classify_right_hand(lm):
    """
    Classify RIGHT hand gesture to determine direction.
    
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

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("Webcam gesture control active (hand classification)")
        print("RIGHT hand gestures:")
        print("  - Fist → LEFT")
        print("  - 1 finger (index) → RIGHT")
        print("  - All fingers open → FORWARD")
        print("  - Fist + thumb out → BACKWARD")
        print("Press L: toggle labels, Q/ESC: quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from webcam")
                break

            flipped = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            action_now = 'stop'  # default if nothing detected

            if results.multi_hand_landmarks and results.multi_handedness:
                h, w = flipped.shape[:2]

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    hand_label = handedness.classification[0].label  # "Left" or "Right"
                    lm = hand_landmarks.landmark

                    # Draw hand skeleton
                    mp_drawing.draw_landmarks(
                        flipped,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    # Show landmark indices if enabled
                    if show_labels:
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)

                            cv2.putText(flipped, str(idx), (x + 4, y - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

                            cv2.putText(flipped, str(idx), (x + 4, y - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                    # RIGHT hand: classify gesture to direction
                    if hand_label == "Right":
                        gesture = classify_right_hand(lm)
                        cv2.putText(
                            flipped, f"Right: {gesture}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2
                        )

                        if gesture == "Forward":
                            action_now = 'forward'
                        elif gesture == "Backward":
                            action_now = 'backward'
                        elif gesture == "Left":
                            action_now = 'left'
                        elif gesture == "Right":
                            action_now = 'right'

            # Debounce: act only when the action changes
            if action_now != last_action:
                if action_now == 'forward':
                    forward()
                elif action_now == 'backward':
                    backward()
                elif action_now == 'left':
                    turn_left()
                elif action_now == 'right':
                    turn_right()
                else:
                    stop()
                last_action = action_now

            # HUD
            hud = {
                'forward': 'FORWARD',
                'backward': 'BACKWARD',
                'left': 'LEFT',
                'right': 'RIGHT',
                'stop': '—'
            }[action_now]

            cv2.putText(flipped, f"Action: {hud}   (L: toggle labels, Q/ESC: quit)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(flipped, f"Action: {hud}   (L: toggle labels, Q/ESC: quit)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Webcam: Hand Gesture Classification", flipped)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('l'):
                show_labels = not show_labels

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
