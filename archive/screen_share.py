import cv2
import numpy as np
import mss
from flask import Flask, Response, request, abort

ACCESS_KEY = "1234"
PORT = 5000

app = Flask(__name__)

def check_key():
    if request.args.get("key", "") != ACCESS_KEY:
        abort(403)

def gen_screen():
    # IMPORTANT: create MSS inside the generator (per-client thread)
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 1 = primary monitor
        while True:
            img = sct.grab(monitor)          # BGRA
            frame = np.array(img)[:, :, :3]  # BGR
            frame = cv2.resize(frame, (960, 540))

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/screen")
def screen():
    check_key()
    return Response(gen_screen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def home():
    check_key()
    return f"""
    <h2>View-only Screen Share</h2>
    <p><a href="/screen?key={ACCESS_KEY}">Open screen stream</a></p>
    """

if __name__ == "__main__":
    # Keep threaded=True ok now, because MSS is created per thread
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True, use_reloader=False)