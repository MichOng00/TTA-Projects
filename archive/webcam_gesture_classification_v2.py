import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

show_labels = True # toggle labels on/off

def classify_right_hand(lm):
    """
    Classify RIGHT hand gesture to determine direction.
    - Fist (all fingers down, thumb not extended) → LEFT
    - 1 finger up (index only) → RIGHT
    - All 4 fingers up (index-pinky extended) → FORWARD
    - Thumb only (fist + thumb extended) → BACKWARD
    - Unrecognized → NONE
    """
    def is_finger_up(tip_idx, pip_idx):
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

    # Gesture patterns
    if four_bits == [0, 0, 0, 0]:
        return "backward" if thumb_up else "left"
    if four_bits == [1, 0, 0, 0]:
        return "right"
    if four_bits == [1, 1, 1, 1]:
        return "forward"
    
    return "none"

last_action = None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        flipped = cv2.flip(frame, 1) # mirror image horizontally
        frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        action_now = "stop" # default if no hands detected
        
        if results.multi_hand_landmarks and results.multi_handedness:
            h, w = flipped.shape[:2]
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                lm = hand_landmarks.landmark
                
                mp_drawing.draw_landmarks(
                    flipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                
                if show_labels:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        # convert position from normalized coordinates
                        x = int(landmark.x * w) 
                        y = int(landmark.y * h)
                        cv2.putText(flipped, str(idx), (x+4, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
                
                # RIGHT hand: classify gesture to direction
                if hand_label == "Right":
                    gesture = classify_right_hand(lm)
                    cv2.putText(flipped, f"Right: {gesture}", (10, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 3, cv2.LINE_AA)
                    
                    if gesture == "forward":
                        action_now = "forward"
                    elif gesture == "backward":
                        action_now = "backward"
                    elif gesture == "left":
                        action_now = "left"
                    elif gesture == "right":
                        action_now = "right"
                    # use the first hand detected
                    if action_now != "stop":
                        break
        
        if action_now != last_action:
            print(action_now)
            last_action = action_now
        
        cv2.imshow("Webcam: Hand Gesture Classification", flipped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_labels = not show_labels

cap.release()
cv2.destroyAllWindows()
