from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/road_damage_final/weights/best.pt")

    metrics = model.val(
        data="datasets/RDD_SPLIT/data.yaml",
        split="test",
        device=0,
        imgsz=640,
        batch=8,
        workers=0,
        plots=True
    )

    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
   
    main()
