import cv2
import numpy as np
import time
import os

USING_UGOT = True

if USING_UGOT:
    from ugot import ugot
    got = ugot.UGOT()
    got.initialize("192.168.1.217")
    got.open_camera()
else:
    cap = cv2.VideoCapture(0)

SAVE_DIR = "./captured"
os.makedirs(SAVE_DIR, exist_ok=True)

counter = 0  # change to one after the last image captured
interval = 1 # how many seconds between captures

print("Press 'q' to stop.")

last_time = time.time()

try:
    while True:
        if USING_UGOT:
            frame = got.read_camera_data()
        else:
            _, frame = cap.read()

        if frame is not None:
            if USING_UGOT:
                nparr = np.frombuffer(frame, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cv2.imshow("UGOT camera", frame)
            else:
                cv2.imshow("Webcam", frame)

            # auto-save
            if time.time() - last_time >= interval:
                filename = f"{SAVE_DIR}/img_{counter:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"saved {filename}")
                counter += 1
                last_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    if not USING_UGOT:
        cap.release()
    cv2.destroyAllWindows()
    print("done.")