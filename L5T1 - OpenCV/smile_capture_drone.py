# Drone Smile Photo Capture
# Uses drone camera to detect smiles and capture photos
import cv2
import numpy as np
import time
from datetime import datetime
import os
from djitellopy import tello

def detect_faces_and_smiles(frame):
    """
    Detect faces and smiles in the frame
    """
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    smile_detected = False

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect smiles in lower half of face
        roi_gray = gray[y + h // 2:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray,
                                               scaleFactor=1.8,
                                               minNeighbors=30,
                                               minSize=(25, 25))

        if len(smiles) > 0:
            smile_detected = True
            for smile in smiles:
                sx, sy, sw, sh = smile
                smile_x = x + sx
                smile_y = y + h // 2 + sy
                cv2.rectangle(frame, (smile_x, smile_y),
                            (smile_x + sw, smile_y + sh), (0, 255, 0), 3)
                cv2.putText(frame, "SMILE!", (smile_x, smile_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

    return frame, smile_detected

def save_photo(frame, folder="drone_smiles"):
    """
    Save photo with timestamp
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"drone_smile_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename

def main():
    # Connect to drone
    tel = tello.Tello()
    tel.connect()
    print(f"Battery: {tel.get_battery()}%")

    tel.streamon()
    # tel.takeoff()
    time.sleep(2)

    image_width, image_height = 720, 480
    smile_count = 0
    last_smile_time = 0
    cooldown = 1.0

    print("Drone Smile Capture - Press 'q' to land and quit")

    try:
        while True:
            frame = tel.get_frame_read().frame
            frame = cv2.resize(frame, (image_width, image_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save clean frame before drawing
            clean_frame = frame.copy()

            frame, smile_detected = detect_faces_and_smiles(frame)

            current_time = time.time()
            if smile_detected and current_time - last_smile_time > cooldown:
                smile_count += 1
                last_smile_time = current_time
                filename = save_photo(clean_frame)  # Save clean frame without HUD
                print(f"Smile #{smile_count} captured: {filename}")

                # Flash green on display frame only
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (image_width, image_height), (0, 255, 0), -1)
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            # Display stats
            cv2.putText(frame, f"Smiles: {smile_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Drone Smile Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # tel.land()
        time.sleep(2)
        tel.streamoff()
        tel.end()
        cv2.destroyAllWindows()
        print(f"Session complete! {smile_count} smiles captured")

if __name__ == "__main__":
    main()