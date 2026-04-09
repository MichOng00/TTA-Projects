from ultralytics import YOLO
model = YOLO("yolo11n.pt")

model.train(data="candle\data.yaml",
            epochs = 50, imgsz = 512)