# Capture images from a local webcam at regular time intervals
# This version replaces the UGOT camera feed with OpenCV webcam capture.
import cv2
import time
import os

SAVE_DIR = "./captured_demo"  # folder for saved webcam images
os.makedirs(SAVE_DIR, exist_ok=True)

# Use the default webcam device (0). Change to 1 or 2 if you have multiple cameras.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check your camera connection and device index.")

counter = 0    # update this if you restart to avoid overwriting previous images
interval = 1   # seconds between auto-saves

print("Webcam capture started. Press 'q' to stop.")
last_time = time.time()

try:
    while True:
        # Read one frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame from webcam")
            continue

        # Display the live webcam feed
        cv2.imshow("Webcam", frame)

        # Save an image every `interval` seconds
        if time.time() - last_time >= interval:
            filename = os.path.join(SAVE_DIR, f"img_{counter:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            counter += 1
            last_time = time.time()

        # Stop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
