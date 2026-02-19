import cv2
import mediapipe as mp
import numpy as np
from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.164")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

cap = cv2.VideoCapture(0)

def lmk_xy(hand_landmarks, idx, w, h):
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype = np.float32)

def euclid(a, b):
    return float(np.linalg.norm(a - b))

def count_fingers_up(hand_landmarks, w, h):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18] # first knuckle
    fingers_up = 0
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = lmk_xy(hand_landmarks, tip_idx, w, h)
        pip = lmk_xy(hand_landmarks, pip_idx, w, h)
        if tip[1] < pip[1]: # compare y-coordinates
            fingers_up += 1

    return fingers_up

def execute_command(fingers_up, speed):
    if fingers_up == 1:
        got.mecanum_move_speed(0, speed) # forward
    elif fingers_up == 2:
        got.mecanum_move_speed(1, speed) # backward
    elif fingers_up == 3:
        got.mecanum_turn_speed(2, speed) # left
    elif fingers_up == 4:
        got.mecanum_turn_speed(3, speed) # right
    else:
        got.mecanum_stop()

try:
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # detect hands
        results = hands.process(frame)

        left_speed = 0
        left_info = ""
        gesture_info = ""
        right_fingers = None
        left_fingers = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness_info.classification[0].label
                confidence = handedness_info.classification[0].score

                # draw landmarks
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
                elif hand_label == "Left":
                    left_fingers = count_fingers_up(hand_landmarks, w, h)

                # display label and confidence
                cv2.putText(frame, f"{hand_label} ({confidence:.2f})",
                            (10,30) if hand_label == "Left" else (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
        # control speed
        if left_fingers:
            left_speed = left_fingers * 10
        else:
            left_speed = 0

        # control direction
        if right_fingers:
            gesture_info = f"Right fingers: {right_fingers}"
            execute_command(right_fingers, left_speed)
        else:
            got.mecanum_stop()

    


        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Q or Esc to break
            break

finally:
    cap.release()
    cv2.destroyAllWindows()