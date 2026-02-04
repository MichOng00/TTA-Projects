import cv2
import mediapipe as mp
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.1.226")
got.open_camera()

mp_draw = mp.solutions.drawing_utils

IDX = {
    # eyes
    # left eye
    "LE_LEFT": 33, "LE_RIGHT": 133, "LE_TOP": 159, "LE_BOTTOM": 145,
    # right eye
    "RE_LEFT": 362, "RE_RIGHT": 263, "RE_TOP": 386, "RE_BOTTOM": 374,

}

def lmk_xy(face_landmarks, idx, w, h):
    lm = face_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype = np.float32)

def euclid(a, b):
    return float(np.linalg.norm(a - b))

def eye_open_ratio(face_landmarks, w, h, is_left=True):
    if is_left:
        top = lmk_xy(face_landmarks, IDX["LE_TOP"], w, h)
        bottom = lmk_xy(face_landmarks, IDX["LE_BOTTOM"], w, h)
        left = lmk_xy(face_landmarks, IDX["LE_LEFT"], w, h)
        right = lmk_xy(face_landmarks, IDX["LE_RIGHT"], w, h)
    else:
        top = lmk_xy(face_landmarks, IDX["RE_TOP"], w, h)
        bottom = lmk_xy(face_landmarks, IDX["RE_BOTTOM"], w, h)
        left = lmk_xy(face_landmarks, IDX["RE_LEFT"], w, h)
        right = lmk_xy(face_landmarks, IDX["RE_RIGHT"], w, h)
    
    vertical = euclid(top, bottom)
    horizontal = euclid(left, right) + 1e-6
    return vertical / horizontal

mp_fm = mp.solutions.face_mesh
face_mesh = mp_fm.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

with face_mesh as fm:
    try:
        while True:
            frame = got.read_camera_data()
            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            h, w = data.shape[:2]
            res = fm.process(frame_rgb)

            if res.multi_face_landmarks:
                for face_landmarks in res.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        data, 
                        face_landmarks,
                        mp_fm.FACEMESH_TESSELATION,
                        landmark_drawing_spec = mp_draw.DrawingSpec(color = (0, 255, 0), thickness = 1, circle_radius = 1),
                        connection_drawing_spec = mp_draw.DrawingSpec(color = (0, 255, 255), thickness = 1)
                    )

                    le_ratio = eye_open_ratio(face_landmarks, w, h, is_left=True)
                    re_ratio = eye_open_ratio(face_landmarks, w, h, is_left=False)

                    EYE_OPEN_THR = 0.2
                    left_open = le_ratio > EYE_OPEN_THR
                    right_open = re_ratio > EYE_OPEN_THR

                    eye_text = f"Eyes: {'Open' if (left_open and right_open) else 'Closed' if (not left_open and not right_open) else 'One closed'}"

                    cv2.putText(data, eye_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if (left_open and right_open) else (0, 165, 255), 2, cv2.LINE_AA)

                # highlight specific points
                # for id, lm in enumerate(face_landmarks.landmark):
                #     h, w, c = data.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)

                #     if id == 1:
                #         cv2.circle(data, (cx, cy), 3, (255, 0, 0), 10)

            cv2.imshow("UGOT camera feed", data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()