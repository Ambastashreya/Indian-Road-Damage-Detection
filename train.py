from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="datasets/RDD_SPLIT/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    name="road_damage_final"
)