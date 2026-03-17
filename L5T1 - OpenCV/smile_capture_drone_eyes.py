# Drone Smile Photo Capture
# Uses drone camera to detect smiles and capture photos
# ADDED: Only captures when BOTH a smile is detected AND both eyes are open
import cv2
import numpy as np
import time
from datetime import datetime
import os
from djitellopy import tello

def detect_faces_and_smiles(frame):
    """
    Detect faces, smiles, and eyes in the frame
    Returns frame, smile_detected, and eyes_open status
    """
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

    # ADDED: Eye detection cascade for checking if both eyes are open
    eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    smile_detected = False
    eyes_open = False  # ADDED: Track if both eyes are detected (open)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ADDED: Detect eyes in upper half of face to check if both eyes are open
        roi_gray_eyes = gray[y:y + h // 2, x:x + w]  # Upper half for eyes
        roi_color_eyes = frame[y:y + h // 2, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray_eyes, 1.1, 4, minSize=(13, 13))

        # ADDED: Check if both eyes are detected (indicating they are open)
        if len(eyes) >= 2:
            eyes_open = True
            # Draw eye rectangles in cyan for visual feedback
            for (ex, ey, ew, eh) in eyes:
                eye_x = x + ex
                eye_y = y + ey
                cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (255, 255, 0), 2)
        else:
            eyes_open = False

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

    return frame, smile_detected, eyes_open

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

            frame, smile_detected, eyes_open = detect_faces_and_smiles(frame)

            current_time = time.time()
            # ADDED: Only capture when BOTH smile is detected AND both eyes are open
            if smile_detected and eyes_open and current_time - last_smile_time > cooldown:
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

            # ADDED: Display eyes status
            eyes_status = "OPEN" if eyes_open else "CLOSED"
            eyes_color = (0, 255, 255) if eyes_open else (0, 0, 255)
            cv2.putText(frame, f"Eyes: {eyes_status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, eyes_color, 2)

            # ADDED: Show capture condition status
            capture_ready = smile_detected and eyes_open
            ready_text = "READY TO CAPTURE" if capture_ready else "Waiting..."
            ready_color = (0, 255, 0) if capture_ready else (100, 100, 100)
            cv2.putText(frame, ready_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ready_color, 2)

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