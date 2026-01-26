from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.220")

import cv2
import numpy as np

got.load_models(["color_recognition"])
got.open_camera()

def color_rec():
    color_info = got.get_color_total_info()

    frame = got.read_camera_data()
    if frame is not None:
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if color_info[0]:
            cv2.putText(data, color_info[0], (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # get centre coordinates, height and width for bounding box
            c_x = color_info[2]
            c_y = color_info[3]
            h = color_info[4]
            w = color_info[5]

            cv2.rectangle(data, (int(c_x - w / 2), int(c_y - h/2)), 
                          (int(c_x + w / 2), int(c_y + h/2)),
                          (0, 0, 255), 2)
            
        else:
            c_x = 320
        
        cv2.imshow("UGOT camera feed", data)
        return color_info[0], c_x

if __name__ == "__main__":
    got.wheelleg_start_balancing()
    try:
        while True:
            color, c_x = color_rec()

            # rotate to scan for a specific color
            if color != "Purple":
                got.wheelleg_turn_speed(turn=3, speed=30)
            else: # keep the color in the centre of the camera view
                if c_x > 330:
                    got.wheelleg_turn_speed(3, 10)
                elif c_x < 310:
                    got.wheelleg_turn_speed(2, 10)
                else:
                    got.wheelleg_stop_balancing()

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()