from ultralytics import YOLO
model = YOLO("yolo11n.pt")
result=model.predict(
    source="test.jpg",
    device=0,
    conf=0.5,
    save=True
    
)