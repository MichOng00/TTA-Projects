import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()

def main():
    got.initialize("192.168.1.217")
    got.open_camera()
    got.load_models(["apriltag_qrcode"])

    try:
        while True:
            tags = got.get_apriltag_total_info()

            frame = got.read_camera_data()
            if not frame:
                print("Failed to grab frame")
                break

            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # # Draw a red rectangle
            # cv2.rectangle(data, (50, 50), (200, 200), (0, 0, 255), 2)

            # # Draw text
            # cv2.putText(data, "Hello OpenCV", (50, 40), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if tags:
                for tag in tags:
                    # Draw a bounding box around recognised apriltag(s)
                    id_num = tag[0]
                    center_x = tag[1]
                    center_y = tag[2]
                    height = tag[3]
                    width = tag[4]
                    cv2.rectangle(data, 
                                  (int(center_x-width//2), int(center_y-height//2)), 
                                  (int(center_x+width//2), int(center_y+height//2)), 
                                  (0, 0, 255), 
                                  2)
                    
                    # Show ID
                    cv2.putText(data, f"{id_num}", (int(center_x-width//2), int(center_y-height//2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    #####################################
                    # Challenge: centralise to a specific ID tag
                    # if tag[0] == 5: # replace with your ID
                    #     apriltag_x = tag[1]
                    #     if apriltag_x > 330:
                    #         got.mecanum_translate_speed(90, 5)
                    #     elif apriltag_x < 310:
                    #         got.mecanum_translate_speed(-90, 5)
                    #     else:
                    #         got.mecanum_stop()
                    #####################################
            

            cv2.imshow("UGOT Camera Feed", data)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()