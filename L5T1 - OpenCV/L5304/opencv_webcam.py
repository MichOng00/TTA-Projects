import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Q or Esc to break
            break

finally:
    cap.release()
    cv2.destroyAllWindows()