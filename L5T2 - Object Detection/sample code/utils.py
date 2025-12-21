import cv2

# Helper: Draw bounding boxes
def draw_detections(frame, results):
    for r in results:
        boxes = r.boxes  # bounding boxes

        for box in boxes:
            # xyxy format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Confidence & label
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label text
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
    return frame