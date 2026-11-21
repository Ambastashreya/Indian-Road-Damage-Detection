from ultralytics import YOLO

model = YOLO("runs/detect/road_damage_final/weights/best.pt")

model.predict(
    source="datasets/RDD_SPLIT/train/images/China_Drone_000636.jpg",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True
)