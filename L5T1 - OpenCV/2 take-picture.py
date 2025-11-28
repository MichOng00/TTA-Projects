"""Minimal example to take a picture with the Tello camera and save it."""
from djitellopy import tello
import time
import cv2

tel = tello.Tello()
tel.connect()
print(f"battery: {tel.get_battery()}")

tel.streamon()
frame_read = tel.get_frame_read()

# tel.takeoff()
time.sleep(1)

frame = frame_read.frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imwrite("picture.png", frame)

time.sleep(1)

# tel.land()
tel.streamoff()
tel.end()