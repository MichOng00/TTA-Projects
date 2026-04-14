import cv2
from ugot import ugot
import time
import numpy as np
from ultralytics import YOLO
from session_01_pretrained_YOLO import draw_detections

got = ugot.UGOT()
got.initialize("192.168.1.111")
got.open_camera()

trained = YOLO("best_coffee.pt")

def find_object():
    try:
        while True:
            frame = got.read_camera_data()
            if frame is not None:
                nparr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                results = trained(img, verbose=False)

                output = draw_detections(img, results)

                # find the highest confidence detection
                area = 10000
                x = 0.5
                max_conf = 0
                found = False
                for r in results:
                    detected = r.boxes.cls.tolist() # class IDs
                    confidences = r.boxes.conf.tolist() # confidence scores
                    xywhn = r.boxes.xywhn.tolist() # normalised coordinates

                    for idx, class_id in enumerate(detected):
                        if class_id == 0:
                            conf = confidences[idx] # get the corresponding confidence
                            if conf > max_conf:
                                max_conf = conf
                                found = True
                                x, y, w, h = xywhn[idx]
                                area = w * h
                    
                    if found:
                        cv2.putText(output, f"Centre: ({x:.3f}, {y:.3f})", (30, 30), 0, 0.6, (0, 255, 0), 2)
                        cv2.putText(output, f"Area: {area:.3f}", (30, 60), 0, 0.6, (0, 255, 0), 2)
                        cv2.putText(output, f"Confidence: {max_conf:.3f}", (30, 90), 0, 0.6, (0, 255, 0), 2)

                        # exercise: centre object in camera view
                        # exercise: move forward / backward if too far / near
                        # Horizontal centering (adjust x-axis rotation)
                        if x > 0.6:
                            got.wheelleg_move_turn(0, 5, 3, 10)  # Turn right
                        elif x < 0.4:
                            got.wheelleg_move_turn(0, 5, 2, 10)  # Turn left
                        else:
                            # Object is centered; approach or stop
                            if area < 0.03:
                                got.wheelleg_move_speed(0, 15) # Move forward
                            elif area < 0.06:
                                got.wheelleg_move_speed(0, 6)  # Move forward
                            elif area > 0.2:
                                got.wheelleg_move_speed(1, 6)  # Move backward
                            else:
                                got.wheelleg_stop_balancing()  # Close enough; stop

                    else:
                        got.wheelleg_turn_speed(3, 10)
                cv2.imshow("YOLO object detection", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        got.wheelleg_stop_balancing()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    got.wheelleg_start_balancing()
    time.sleep(1)
    find_object()