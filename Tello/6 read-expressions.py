import cv2
import mediapipe as mp
import numpy as np
from djitellopy import tello
import math

tel = tello.Tello()
tel.connect()

print(f"Battery: {tel.get_battery()}")

tel.streamon()

frame_read = tel.get_frame_read()

# Initialize MediaPipe Face Detection
mp_draw = mp.solutions.drawing_utils

# Mediapipe Face Mesh landmark indices we need
# (https://web.archive.org/web/20230324031132/https://google.github.io/mediapipe/solutions/face_mesh.html)
IDX = {
    # Left eye (subject's left)
    "LE_LEFT": 33, "LE_RIGHT": 133, "LE_TOP": 159, "LE_BOTTOM": 145,
    # Right eye (subject's right)
    "RE_LEFT": 362, "RE_RIGHT": 263, "RE_TOP": 386, "RE_BOTTOM": 374,
    # Mouth corners & mid lips (inner)
    "M_LEFT": 61, "M_RIGHT": 291, "LIP_TOP": 13, "LIP_BOTTOM": 14,
}

def lmk_xy(face_landmarks, idx, w, h):
    lm = face_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

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
    return vertical / horizontal  # EAR-like ratio

def smile_metrics(face_landmarks, w, h):
    left = lmk_xy(face_landmarks, IDX["M_LEFT"], w, h)
    right = lmk_xy(face_landmarks, IDX["M_RIGHT"], w, h)
    lip_top = lmk_xy(face_landmarks, IDX["LIP_TOP"], w, h)
    lip_bottom = lmk_xy(face_landmarks, IDX["LIP_BOTTOM"], w, h)

    mouth_width = euclid(left, right)
    mouth_height = euclid(lip_top, lip_bottom)

    # Normalize by inter-ocular (outer) width for scale invariance
    le_outer = lmk_xy(face_landmarks, IDX["LE_LEFT"], w, h)
    re_outer = lmk_xy(face_landmarks, IDX["RE_RIGHT"], w, h)
    eye_outer_width = euclid(le_outer, re_outer) + 1e-6

    width_norm = mouth_width / eye_outer_width
    height_norm = mouth_height / eye_outer_width

    # "Corner lift": corners higher (smaller y) than upper lip when smiling
    corners_y = (left[1] + right[1]) * 0.5
    corner_lift = (lip_top[1] - corners_y) / eye_outer_width  # larger => more upturned

    return width_norm, height_norm, corner_lift
# -------------------- End added helpers --------------------

# Initialize Face Mesh model
mp_fm = mp.solutions.face_mesh
face_mesh = mp_fm.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,  # Track eye irises
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

with face_mesh as fm:
    while True:
        frame = frame_read.frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]  # <-- Added
        res = fm.process(frame_rgb)
            
        if res.multi_face_landmarks:
            for face_landmarks in res.multi_face_landmarks:
                # Draw all landmarks (existing)
                mp_draw.draw_landmarks(
                    frame_rgb,
                    face_landmarks,
                    mp_fm.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1)
                )

                # -------------------- Added: Eyes open/closed --------------------
                # Compute simple EAR-style ratios for both eyes
                le_ratio = eye_open_ratio(face_landmarks, w, h, is_left=True)
                re_ratio = eye_open_ratio(face_landmarks, w, h, is_left=False)

                # Thresholds: tune if needed (typical range ~0.18 closed, ~0.26 open)
                EYE_OPEN_THR = 0.23
                left_open = le_ratio > EYE_OPEN_THR
                right_open = re_ratio > EYE_OPEN_THR

                # -------------------- Added: Smile detection --------------------
                width_norm, height_norm, corner_lift = smile_metrics(face_landmarks, w, h)
                # Heuristic: smiling if mouth is relatively wide and corners are lifted a bit
                SMILE_WIDTH_THR = 0.6
                CORNER_LIFT_THR = -0.04
                smiling = (width_norm > SMILE_WIDTH_THR) and (corner_lift > CORNER_LIFT_THR)
                # better to involve eyebrows
                # frowning = (width_norm < SMILE_WIDTH_THR) and (corner_lift < -0.07)

                # -------------------- Added: On-screen HUD --------------------
                eye_text = f"Eyes: {'Open' if (left_open and right_open) else 'Closed' if (not left_open and not right_open) else 'One closed'}"
                smile_text = f"Smile: {'Yes' if smiling else 'No'}"
                # frown_text = f"Frown: {'Yes' if frowning else 'No'}"
                debug_text = f"(LE:{le_ratio:.2f} RE:{re_ratio:.2f})  width:{width_norm:.2f} lift:{corner_lift:.3f}"

                cv2.putText(frame_rgb, eye_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if (left_open and right_open) else (0,165,255), 2, cv2.LINE_AA)
                cv2.putText(frame_rgb, smile_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if smiling else (0,165,255), 2, cv2.LINE_AA)
                # cv2.putText(frame_rgb, frown_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if frowning else (0,165,255), 2, cv2.LINE_AA)
                # (Optional) tiny debug line; comment out if noisy
                cv2.putText(frame_rgb, debug_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
                cv2.putText(frame_rgb, f"{width_norm:.2f}, {corner_lift:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        # Display frame
        cv2.imshow("Facial expressions", frame_rgb)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    tel.streamoff()
    tel.end()
    cv2.destroyAllWindows()