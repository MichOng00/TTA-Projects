import cv2
import numpy as np
import time

from ultralytics import YOLO
from utils import draw_detections

from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.217")

def line_follow_camera():
    got.open_camera()

    got.set_track_recognition_line(line_type = 0)

    line_info = got.get_single_track_total_info()  # list: [offset, type, x, y]
    offset = line_info[0]
    line_type = line_info[1]

    try:
        while True:
            frame = got.read_camera_data()
            if frame is not None:
                nparr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Run YOLO detection
                results = model(img, verbose=False)

                # Draw output
                output = draw_detections(img, results)

                # Show
                cv2.imshow("YOLO Detection - With Custom Object", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            line_info = got.get_single_track_total_info()
            offset = line_info[0]
            line_type = line_info[1]

            if line_type != 1: # no line, or intersection, or crossroads
                return line_type, results

            degrees = int(offset / 4)
            got.mecanum_move_xyz(0, 20, degrees)
            time.sleep(0.1)

    finally:
        got.mecanum_stop()

def ap_approach():
    while True:
        AP_info = got.get_apriltag_total_info()
        if AP_info:
            x_coord = AP_info[0][1]
            dist = AP_info[0][6]
            if x_coord < 290:
                 got.mecanum_move_xyz(-3, 3, 0)
            elif x_coord > 350:
                got.mecanum_move_xyz(3, 3, 0)
            elif dist > 0.12:
                got.mecanum_move_xyz(0, 3, 0)
            else:
                got.mecanum_stop()
                break
        else:
            got.mecanum_stop()
    print("Stopped.")

def pickup_ap():
    got.mechanical_clamp_release()
    time.sleep(1)
    got.mechanical_joint_control(0, 0, -70, 800) #down - for apriltag
    time.sleep(1)
    got.mechanical_clamp_close()
    time.sleep(2)
    got.mechanical_joint_control(0, 30, -50, 800) #up

def put_ap():
    got.mechanical_joint_control(-90, 30, -50, 800)
    time.sleep(1)
    got.mechanical_joint_control(-90, -20, -30, 800)
    time.sleep(1)
    got.mechanical_clamp_release()
    time.sleep(2)

if __name__ == "__main__":
    model = YOLO("best_coffee.pt")

    got.load_models(["line_recognition", "apriltag_qrcode"])

    num_intersections = 0

    try:
        while num_intersections < 2:
            line_type, results = line_follow_camera()
            if line_type == 2:
                num_intersections += 1
                for r in results:
                    # ['candle', 'coffee', 'coke', 'token']

                    print(r.boxes)
                    detected = r.boxes.cls.tolist()
                    if 0 in detected: # candle
                        got.mecanum_turn_speed_times(2, 40, 20, 2)
                        got.mecanum_translate_speed_times(0, 10, 20, 1)
                    elif 3 in detected: # token
                        ap_approach()
                        pickup_ap()
                        got.mecanum_turn_speed_times(2, 40, 180, 2)
                    else:
                        got.mecanum_turn_speed_times(3, 40, 20, 2)
    finally:
        got.mecanum_stop()
        cv2.destroyAllWindows()