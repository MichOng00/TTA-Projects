import cv2
import time
from datetime import datetime
import os
from djitellopy import tello

def detect_faces_and_smiles(frame):
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    smile_detected = False

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        lower_face = gray[y + h // 2: y + h, x:x+w]
        smiles = smile_cascade.detectMultiScale(lower_face, 1.8, 25, minSize=(25, 25))

        if len(smiles) > 0:
            smile_detected = True
            for smile in smiles:
                sx, sy, sw, sh = smile
                smile_x = x + sx
                smile_y = y + h // 2 + sy
                cv2.rectangle(frame, (smile_x, smile_y), (smile_x + sw, smile_y + sh), (0, 255, 0), 2)

    return frame, smile_detected

def save_photo(frame, folder="smiles"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"smile_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename

if __name__ == "__main__":
    tel = tello.Tello()
    tel.connect()
    print(f"battery: {tel.get_battery()}")
    tel.streamon()

    image_width, image_height = 720, 480
    smile_count = 0
    last_smile_time = 0
    cooldown = 3

    try:
        while True:
            frame = tel.get_frame_read().frame
            frame = cv2.resize(frame, (image_width, image_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame, smile_detected = detect_faces_and_smiles(frame)
            current_time = time.time()

            if smile_detected and current_time - last_smile_time > cooldown:
                smile_count += 1
                last_smile_time = current_time
                filename = save_photo(frame)

                # flash green
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (image_width, image_height), (0, 255, 0), -1)
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            cv2.putText(frame, f"Smiles: {smile_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0, 255, 0), 2)
            # cv2.putText(frame, f"Cooldown: {max(-current_time + last_smile_time + cooldown, 0):.2f}", 
            #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            time_to_pic = int(-current_time + last_smile_time + cooldown)
            if time_to_pic < 0:
                time_to_pic = "ready"
            cv2.putText(frame, f"Cooldown: {time_to_pic}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Drone camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tel.streamoff()
        tel.end()
        cv2.destroyAllWindows()