# to switch models, replace all wheelleg with balance (or vice versa)
from ugot import ugot
import time
import numpy as np
import cv2

# Initialize UGOT
got = ugot.UGOT()
got.initialize('192.168.1.136')
got.open_camera()
got.load_models(["line_recognition"])
got.set_track_recognition_line(0)

got.balance_start_balancing()
# got.wheelleg_set_chassis_height(3)
time.sleep(2)
try:
    while True:
        frame = got.read_camera_data()
        if frame is not None:
            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow("UGOT Camera", data)
            cv2.waitKey(1)

        offset, type, x, y = got.get_single_track_total_info()
        rotation_speed = abs(int(offset * 0.2))

        if offset > 0:
            direction = 2
        else:
            direction = 3
        got.balance_move_turn(0, 10, direction, rotation_speed)
finally:
    got.balance_stop_balancing()
    cv2.destroyAllWindows()