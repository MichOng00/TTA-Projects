# see latest image analysis at http://192.168.1.69:5000/status
# takes a few minutes to analyse
import cv2
import requests
import threading
import time
import os

from datetime import datetime
from gpiozero import DigitalInputDevice
from flask import Flask, Response, jsonify

# ---------------- Configuration ----------------

app = Flask(__name__)

CAMERA_INDEX = 0
SENSOR_PIN = 24

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "minicpm-v"

SAVE_DIR = "pictures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Sensor is LOW when an obstacle is detected,
# matching the example provided.
sensor = DigitalInputDevice(SENSOR_PIN, pull_up=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

camera_lock = threading.Lock()
status_lock = threading.Lock()
analysis_lock = threading.Lock()

status = {
    "obstacle_detected": False,
    "image": None,
    "analysis": "Waiting for obstacle...",
    "updated_at": None
}

# ---------------- Camera ----------------

def capture_image():
    """Capture one frame safely while Flask streams video."""
    with camera_lock:
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read camera")
            return None

        filename = os.path.join(
            SAVE_DIR,
            datetime.now().strftime("obstacle_%Y%m%d_%H%M%S_%f.jpg")
        )

        if not cv2.imwrite(filename, frame):
            print("ERROR: Could not save image")
            return None

    print(f"Image saved: {filename}")
    return filename


def generate_frames():
    while True:
        with camera_lock:
            success, frame = cap.read()

        if not success:
            time.sleep(0.1)
            continue

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------- MiniCPM-V image analysis ----------------

def analyze_image(filename):
    # Only run one vision request at a time.
    with analysis_lock:
        try:
            with open(filename, "rb") as image_file:
                image_bytes = image_file.read()

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a visual monitoring assistant. "
                                "Describe only what is visible. "
                                "Do not guess identities or claim certainty "
                                "when the image is unclear."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "An obstacle sensor has just detected "
                                "something. Describe the visible scene, "
                                "identify any visible object near the "
                                "camera, and mention any apparent hazard. "
                                "Keep the answer concise."
                            ),
                            "images": [
                                __import__("base64").b64encode(
                                    image_bytes
                                ).decode("utf-8")
                            ]
                        }
                    ],
                    "stream": False
                },
                timeout=180
            )

            response.raise_for_status()
            result = response.json()["message"]["content"]

            print("\n--- MiniCPM-V analysis ---")
            print(result)
            print("--------------------------\n")

            with status_lock:
                status["analysis"] = result
                status["updated_at"] = datetime.now().isoformat(
                    timespec="seconds"
                )

        except Exception as error:
            message = f"Image analysis failed: {error}"
            print(message)

            with status_lock:
                status["analysis"] = message
                status["updated_at"] = datetime.now().isoformat(
                    timespec="seconds"
                )


# ---------------- Sensor monitoring ----------------

def obstacle_detected():
    """Capture immediately on a new obstacle event."""
    filename = capture_image()

    if filename is None:
        return

    with status_lock:
        status["obstacle_detected"] = True
        status["image"] = filename
        status["analysis"] = "Image captured; AI is analysing..."
        status["updated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )

    # Do not block sensor monitoring while the model runs.
    threading.Thread(
        target=analyze_image,
        args=(filename,),
        daemon=True
    ).start()


def monitor_sensor():
    was_detected = False
    candidate_state = None
    candidate_since = time.monotonic()

    while True:
        # Based on your example: LOW means obstacle detected.
        detected = not sensor.is_active
        now = time.monotonic()

        # Require the new state to remain stable for 150 ms
        # to reduce false triggers from electrical noise.
        if detected != was_detected:
            if candidate_state != detected:
                candidate_state = detected
                candidate_since = now

            elif now - candidate_since >= 0.15:
                was_detected = detected

                if detected:
                    print("Obstacle detected!")
                    obstacle_detected()
                else:
                    print("No obstacle detected")

                    with status_lock:
                        status["obstacle_detected"] = False
                        status["updated_at"] = datetime.now().isoformat(
                            timespec="seconds"
                        )

                candidate_state = None
        else:
            candidate_state = None

        time.sleep(0.02)


@app.route("/status")
def get_status():
    with status_lock:
        return jsonify(status.copy())


# ---------------- Start application ----------------

if __name__ == "__main__":
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAMERA_INDEX.")

    threading.Thread(
        target=monitor_sensor,
        daemon=True
    ).start()

    print("Camera stream: http://<PI-IP>:5000/")
    print("Sensor status: http://<PI-IP>:5000/status")
    print("Waiting for obstacle...")

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            debug=False,
            use_reloader=False
        )
    finally:
        sensor.close()
        cap.release()