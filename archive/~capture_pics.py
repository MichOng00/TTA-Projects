# Save images from UGOT video feed at regular time intervals
from ugot import ugot
import cv2
import numpy as np
import time
import os

SAVE_DIR = "AIMS_cubes" # changed folder for demo purposes
os.makedirs(SAVE_DIR, exist_ok=True)

got = ugot.UGOT()
got.initialize("192.168.1.180")
got.open_camera()
got.transform_set_chassis_height(7)

counter = 0    # with multiple runs, change this to one after the last captured image name to avoid overwriting images
interval = 3   # seconds between captures

print("Auto-capturing images. Press 'q' to stop.")

last_time = time.time()

try:
    while True:
        frame = got.read_camera_data()
        if frame is not None:
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow("UGOT Camera", img)

            # Auto-save
            if time.time() - last_time >= interval:
                filename = f"{SAVE_DIR}/pink_{counter:04d}.jpg"
                cv2.imwrite(filename, img)
                print(f"Saved: {filename}")
                counter += 1
                last_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cv2.destroyAllWindows()
    print("Done.")
