"""Keyboard control of Tello drone flight."""
from djitellopy import tello
import cv2

tel = tello.Tello()
tel.connect()
print(f"battery: {tel.get_battery()}")

try:

    tel.takeoff()

    while True:
        frame_read = tel.get_frame_read()
        img = frame_read.frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Drone camera", img)
    
        key = cv2.waitKey(1) & 0xff
        if key == 27: # ESC
            break
        elif key == ord('w'):
            tel.move_forward(30)
        elif key == ord('s'):
            tel.move_back(30)
        elif key == ord('a'):
            tel.move_left(30)
        elif key == ord('d'):
            tel.move_right(30)
        elif key == ord('l'):
            tel.rotate_clockwise(30)
        elif key == ord('j'):
            tel.rotate_counter_clockwise(30)
        elif key == ord('i'):
            tel.move_up(30)
        elif key == ord('k'):
            tel.move_down(30)

finally:
    tel.land()
    tel.end()
    cv2.destroyAllWindows()