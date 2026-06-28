import cv2
import mediapipe as mp
from ugot import ugot
import numpy as np

got = ugot.UGOT()
got.initialize('192.168.88.1')
got.open_camera()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Optional: press 'l' to toggle labels on/off during runtime
show_labels = True
pointing_active_prev = False  # debounce state

def forward():
    print("go forward")

def stop():
    print("stop")

def is_index_pointing_up(landmarks):
    """
    Returns True if: index finger is clearly extended upward AND
    middle/ring/pinky are curled (not extended).
    Thumb is ignored (neutral allowed).
    """
    # Convenience accessors
    def y(i): return landmarks[i].y

    # Monotonic "extended" test for index (tip above others)
    index_extended = (y(8) < y(7) < y(6) < y(5))

    # "Curled" tests for the other fingers: tip lower than PIP and MCP
    middle_curled = (y(12) > y(10) and y(12) > y(9))
    ring_curled   = (y(16) > y(14) and y(16) > y(13))
    pinky_curled  = (y(20) > y(18) and y(20) > y(17))

    # Optional: disallow a straight-up thumb (rarely vertical; usually sideways).
    # If you want to be strict, uncomment this:
    # thumb_not_up = (y(4) > y(2))
    # return index_extended and middle_curled and ring_curled and pinky_curled and thumb_not_up

    return index_extended and middle_curled and ring_curled and pinky_curled

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

        # Process with MediaPipe
        results = hands.process(frame_rgb)

        pointing_active_now = False

        # Draw hand landmarks + numeric labels
        if results.multi_hand_landmarks:
            h, w = flipped.shape[:2]

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the default connections/landmarks
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

                # Gesture detection: only index pointing up
                if is_index_pointing_up(hand_landmarks.landmark):
                    pointing_active_now = True

        # Debounce: trigger on rising edge only
        if pointing_active_now and not pointing_active_prev:
            forward()
        elif not pointing_active_now and pointing_active_prev:
            stop()
        pointing_active_prev = pointing_active_now

        # HUD
        status = "INDEX UP" if pointing_active_now else "—"
        cv2.putText(flipped, f"Gesture: {status}   (L: toggle labels, Q/ESC: quit)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(flipped, f"Gesture: {status}   (L: toggle labels, Q/ESC: quit)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("MediaPipe Hands (indices + pointing-up gesture)", flipped)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('l'):
            show_labels = not show_labels

cv2.destroyAllWindows()
