from ultralytics import YOLO

def main():
    model = YOLO(
        "runs/detect/road_damage_final/weights/best.pt"
    )

    model.train(
        data="datasets/RDD_SPLIT/data.yaml",
        epochs=40,
        patience=5,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        name="road_damage_extended"
    )

if __name__ == "__main__":
    main()