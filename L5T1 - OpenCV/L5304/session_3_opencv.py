from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.131")
import cv2
import numpy as np

got.load_models(["apriltag_qrcode"])
got.open_camera()

try:
    while True:
        frame = got.read_camera_data()
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        tags = got.get_apriltag_total_info() # list of lists
        for tag in tags:
            id_num = tag[0]
            center_x = tag[1]
            center_y = tag[2]
            height = tag[3]
            width = tag[4]
            # draw bounding box
            cv2.rectangle(data, 
                            (int(center_x-width//2), int(center_y-height//2)), 
                            (int(center_x+width//2), int(center_y+height//2)), 
                            (255, 0, 0), 2)
            
            # centralise apriltag
            if center_x > 330:
                got.mecanum_translate_speed(90, 5)
            elif center_x < 310:
                got.mecanum_translate_speed(-90, 5)
            else:
                got.mecanum_stop()

        cv2.imshow("UGOT camera feed", data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    got.mecanum_stop()
    cv2.destroyAllWindows()