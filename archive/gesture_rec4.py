import cv2
import mediapipe as mp
from ugot import ugot
import numpy as np
import math

got = ugot.UGOT()
got.initialize('192.168.88.1')
got.open_camera()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

show_labels = True

# Debounce: remember last action ('forward', 'backward', 'left', 'right', 'stop')
last_action = None

def forward():
    print("go forward")
    got.mecanum_move_speed(0, 30)

def backward():
    print("go back")
    got.mecanum_move_speed(1, 30)

def turn_left():
    print("go left")
    got.mecanum_turn_speed(2, 45)

def turn_right():
    print("go right")
    got.mecanum_turn_speed(3, 45)

def stop():
    print("stop")
    got.mecanum_stop()

def only_index_extended(landmarks, extended_thresh=0.2, curl_thresh=0.3):
    """
    Detects: index finger extended, others curled, regardless of pointing direction.
    - extended_thresh: min distance between tip and MCP for "extended"
    - curl_thresh: max distance between tip and MCP for "curled"
    """
    def dist(a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

    # Distances from MCP to tip
    index_len = dist(landmarks[5], landmarks[8])
    middle_len = dist(landmarks[9], landmarks[12])
    ring_len   = dist(landmarks[13], landmarks[16])
    pinky_len  = dist(landmarks[17], landmarks[20])

    index_extended = index_len > extended_thresh
    middle_curled  = middle_len < curl_thresh
    ring_curled    = ring_len < curl_thresh
    pinky_curled   = pinky_len < curl_thresh

    return index_extended and middle_curled and ring_curled and pinky_curled


def direction_from_index(landmarks, angle_deadzone_deg=30, min_reach=0.02):
    """
    Return 'up'/'down'/'left'/'right' if index points clearly in that direction,
    else None.

    - angle is computed from MCP(5) -> TIP(8)
    - coordinate fix: y grows downward in images, so use vy = (mcp_y - tip_y)
    - bins:
        right: |angle| <= 30
        up:    60 <= angle <= 120
        left:  angle >= 150 or angle <= -150
        down:  -120 <= angle <= -60
    - require minimum reach (tip far enough from MCP) to avoid jitter.
    """
    tip = landmarks[8]
    mcp = landmarks[5]

    # Work in normalized coords
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    # Flip y for conventional math (up is positive)
    vx, vy = dx, (mcp.y - tip.y)

    # Reach (normalized): how extended the finger is
    reach = math.hypot(vx, vy)
    if reach < min_reach:
        return None

    angle = math.degrees(math.atan2(vy, vx))  # 0=right, 90=up, -90=down, 180/-180=left

    # Direction bins
    if -angle_deadzone_deg <= angle <= angle_deadzone_deg:
        return 'left'
    if 60 <= angle <= 120:
        return 'up'
    if angle >= 150 or angle <= -150:
        return 'right'
    if -120 <= angle <= -60:
        return 'down'
    return None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        frame = got.read_camera_data()
        if not frame:
            break
        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        flipped = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        action_now = 'stop'  # default if nothing detected

        if results.multi_hand_landmarks:
            h, w = flipped.shape[:2]

            # Use the first good hand that matches our gesture
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    flipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # Show landmark indices
                if show_labels:
                    # Label each landmark with its MediaPipe index (0..20)
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        cv2.putText(flipped, str(idx), (x + 4, y - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

                        cv2.putText(flipped, str(idx), (x + 4, y - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # Gesture: only index extended
                if only_index_extended(hand_landmarks.landmark):
                    dir_ = direction_from_index(hand_landmarks.landmark)
                    if dir_ == 'up':
                        action_now = 'forward'
                    elif dir_ == 'down':
                        action_now = 'backward'
                    elif dir_ == 'left':
                        action_now = 'left'
                    elif dir_ == 'right':
                        action_now = 'right'
                # Use the first qualifying hand
                if action_now != 'stop':
                    break

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
            'forward': 'INDEX: UP',
            'backward': 'INDEX: DOWN',
            'left': 'INDEX: LEFT',
            'right': 'INDEX: RIGHT',
            'stop': '—'
        }[action_now]

        cv2.putText(flipped, f"Gesture: {hud}   (L: toggle labels, Q/ESC: quit)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(flipped, f"Gesture: {hud}   (L: toggle labels, Q/ESC: quit)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("MediaPipe Hands (directional pointing gesture)", flipped)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('l'):
            show_labels = not show_labels

cv2.destroyAllWindows()
