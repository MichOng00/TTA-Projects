import cv2
import mediapipe as mp
from ugot import ugot
import numpy as np

got = ugot.UGOT()
got.initialize('192.168.1.29')
got.open_camera()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Optional: press 'l' to toggle labels on/off during runtime
show_labels = True

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

                if show_labels:
                    # Label each landmark with its MediaPipe index (0..20)
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        cv2.putText(flipped, str(idx), (x + 4, y - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                        
                        cv2.putText(flipped, str(idx), (x + 4, y - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Optional on-screen hint
        cv2.putText(flipped, "Press 'L' to toggle labels, 'Q' or ESC to quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(flipped, "Press 'L' to toggle labels, 'Q' or ESC to quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("MediaPipe Hands (with Joint Indices)", flipped)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('l'):
            show_labels = not show_labels

cv2.destroyAllWindows()
