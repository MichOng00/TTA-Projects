import cv2
import mediapipe as mp
from ugot import ugot
import numpy as np
import math
cap = cv2.VideoCapture(0) # webcam
got = ugot.UGOT()
got.initialize("10.109.101.208")
got.open_camera()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

show_labels = True # toggle labels on/off
def only_index_extended(landmarks, extended_thresh=0.2, curl_thresh=0.3):
    def dist(a,b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
    index_len = dist(landmarks[5], landmarks[8])
    middle_len = dist(landmarks[9], landmarks[12])
    ring_len = dist(landmarks[13], landmarks[16])
    pinky_len = dist(landmarks[17], landmarks[20])
    index_extended = index_len > extended_thresh
    middle_curled = middle_len < curl_thresh
    ring_curled = ring_len < curl_thresh
    pinky_curled = pinky_len < curl_thresh
    return index_extended and middle_curled and ring_curled and pinky_curled

def direction_from_index(landmarks, deadzone_deg=30, min_reach=0.02):
    tip = landmarks[8]
    mcp = landmarks[5]
    dx = tip.x - mcp.x
    dy = mcp.y - tip.y
    reach = math.hypot(dx, dy)
    if reach < min_reach:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    if -deadzone_deg <= angle <= deadzone_deg:
        return "left"
    if 60 <= angle <= 120:
        return "up"
    if angle >= 150 or angle <= -150:
        return "right"
    if -120 <= angle <= -60:
        return "down"
    return None # no clear direction
last_action = None
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        # frame = got.read_camera_data()
        # if not frame:
        #     continue
        # nparr = np.frombuffer(frame, np.uint8)
        # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        ret, frame = cap.read()
        flipped = cv2.flip(frame, 1) # mirror image horizontally
        frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        action_now = "stop" # default if no hands detected
        if results.multi_hand_landmarks:
            h, w = flipped.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    flipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                if show_labels:
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        # convert position from normalized coordinates
                        x = int(lm.x * w) 
                        y = int(lm.y * h)
                        cv2. putText(flipped, str(idx), (x+4, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
                if only_index_extended(hand_landmarks.landmark):
                    cv2.putText(flipped, "extended", (10, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 3, cv2.LINE_AA)
                    direction = direction_from_index(hand_landmarks.landmark)
                    if direction == "up":
                        action_now = "forward"
                    elif direction == "down":
                        action_now = "backward"
                    elif direction == "left":
                        action_now = "left"
                    elif direction == "right":
                        action_now = "right"
                    # use the first hand detected
                    if action_now != "stop":
                        break
        if action_now != last_action:
            if action_now == "forward":
                got.mecanum_move_speed(0, 30)
            elif action_now == "backward":
                got.mecanum_move_speed(1, 30)
            elif action_now == "left":
                got.mecanum_turn_speed(3, 45)
            elif action_now == "right":
                got.mecanum_turn_speed(2, 45)
            else:
                got.mecanum_stop()
            last_action = action_now
        cv2.imshow("Mediapipe hands", flipped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_labels = not show_labels
    cv2.destroyAllWindows()
    cap.release() # release webcam for other apps