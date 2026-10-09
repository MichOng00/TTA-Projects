import cv2
from flask import Flask, Response
from gpiozero import Button
from datetime import datetime
import os

app = Flask(__name__)
cap = cv2.VideoCapture(0)
button = Button(18)

# Folder to save pictures
SAVE_DIR = "pictures"
os.makedirs(SAVE_DIR, exist_ok=True)

def take_picture():
    success, frame = cap.read()

    if success:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"picture_{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        print(f"Picture saved: {filename}")
    else:
        print("Failed to capture image")

# Take picture when button is pressed
button.when_pressed = take_picture

def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cap.release()