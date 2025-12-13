import cv2
import mediapipe as mp
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.1.217")
got.open_camera()

# Initialize MediaPipe Face Detection
mp_draw = mp.solutions.drawing_utils

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
        frame = got.read_camera_data()
        if not frame:
            break
        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(frame_rgb)
            
        if res.multi_face_landmarks:
            for face_landmarks in res.multi_face_landmarks:
                # Draw all landmarks
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_fm.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1)
                )
                
                # Highlight specific points (e.g., eyes, nose, mouth)
                for id, lm in enumerate(face_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Example: Highlight nose tip (landmark 1)
                    if id == 1:
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)  # Red circle

                    # Example: Highlight left eye iris (landmark 473)
                    if id == 473:
                        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)  # Blue circle'''

        # Display frame
        cv2.imshow("Face mesh", frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()