import cv2
import numpy as np
from ultralytics import YOLO

from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.217")
got.open_camera()

# load pretrained model
model = YOLO("yolo11n.pt")

# helper function to draw bounding boxes
def draw_detections(frame, results):
    for r in results:
        boxes = r.boxes 
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            conf = float(box.conf[0]) # confidence
            cls_id = int(box.cls[0]) # class id
            label = r.names[cls_id]  # class name

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 8), 0, 0.6, (0, 255, 0), 2)

    return frame

if __name__ == "__main__":
    while True:
        frame = got.read_camera_data()
        if frame is not None:
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            results = model(img, verbose = False)
            output = draw_detections(img, results)

            cv2.imshow("YOLO detection", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()