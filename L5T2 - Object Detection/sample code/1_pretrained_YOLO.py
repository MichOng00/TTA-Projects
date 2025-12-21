import cv2
import numpy as np

from ultralytics import YOLO

from utils import draw_detections

from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.217")

# Load pretrained model from YOLO 
model = YOLO("yolo11n.pt")

if __name__ == "__main__":
    # Visualize bounding boxes with live video feed
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
            cv2.imshow("YOLO Detection", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()