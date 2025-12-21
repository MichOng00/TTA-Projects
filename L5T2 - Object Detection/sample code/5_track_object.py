import cv2
from ugot import ugot
import time
import numpy as np
from ultralytics import YOLO
from utils import draw_detections

# ============================================================================
# UGOT INITIALIZATION
# ============================================================================
got = ugot.UGOT()
got.initialize("192.168.1.230")  # Change IP based on robot
got.open_camera()

# Load pre-trained custom YOLO model (must be trained on your own dataset)
trained = YOLO("../IMDA/best_coffee.pt")

def find_object():
    """
    Object detection and approach loop using YOLO.
    
    Continuously:
    1. Reads camera frames from UGOT
    2. Runs YOLO detection to find objects
    3. Tracks the highest confidence detection
    4. Centers and approaches the object
    
    Movement Strategy:
    - If object x > 0.6 (right side): turn right while moving forward
    - If object x < 0.4 (left side): turn left while moving forward
    - If object x ≈ 0.5 (centered): move forward or stop when close enough
    - If area < 0.06 (far): move forward
    - If area ≥ 0.06 (close enough): stop
    - If no object detected: turn right to scan
    
    Exit: Press 'Q' to quit.
    """
    print("Object detection started.")
    try:
        while True:
            frame = got.read_camera_data()
            if frame is not None:
                # Decode camera frame from bytes
                nparr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Run YOLO detection
                results = trained(img, verbose=False)

                # Draw bounding boxes
                output = draw_detections(img, results)

                # Find highest confidence detection
                area = 1000
                x = 0.5
                max_conf = 0.0
                max_idx = -1

                for r in results:
                    detected = r.boxes.cls.tolist()  # Class IDs
                    confidences = r.boxes.conf.tolist()  # Confidence scores
                    xywhn = r.boxes.xywhn.tolist()  # Normalized coords [x, y, w, h]

                    for idx, cls_id in enumerate(detected):
                        if cls_id == 0:  # Candle class (change if different)
                            conf = confidences[idx]
                            if conf > max_conf:
                                max_conf = conf
                                max_idx = idx
                                x, y, w, h = xywhn[idx]
                                area = w * h

                # Control robot based on object detection result
                if max_idx != -1:  # Object found
                    # Display detection info
                    cv2.putText(output, f"Centre: ({x:.3f}, {y:.3f})", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(output, f"Area: {area:.3f}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(output, f"Confidence: {max_conf:.3f}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Horizontal centering (adjust x-axis rotation)
                    if x > 0.6:
                        got.balance_move_turn(0, 5, 3, 10)  # Turn right
                    elif x < 0.4:
                        got.balance_move_turn(0, 5, 2, 10)  # Turn left
                    else:
                        # Object is centered; approach or stop
                        if area < 0.03:
                            got.balance_move_speed(0, 15) # Move forward
                        elif area < 0.06:
                            got.balance_move_speed(0, 6)  # Move forward
                        elif area > 0.2:
                            got.balance_move_speed(1, 6)  # Move backward
                        else:
                            got.balance_stop_balancing()  # Close enough; stop

                else:  # No object detected
                    got.balance_turn_speed(3, 10)  # Scan by turning right

                # Display live camera feed
                cv2.imshow("YOLO Object Detection", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        got.balance_stop_balancing()
        cv2.destroyAllWindows()
        print("❌ Object detection stopped.")

if __name__ == "__main__":
    got.balance_start_balancing()
    time.sleep(1)
    find_object()