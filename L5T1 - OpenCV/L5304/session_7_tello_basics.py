# pip install djitellopy
from djitellopy import tello
import time
import cv2

tel = tello.Tello()
tel.connect()
print(f"battery: {tel.get_battery()}")

# open camera
tel.streamon()
frame_read = tel.get_frame_read()

tel.takeoff()
time.sleep(1)

frame = frame_read.frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imwrite("picture.png", frame)

tel.streamoff()
tel.land()