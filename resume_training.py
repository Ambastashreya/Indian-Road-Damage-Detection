from ultralytics import YOLO

def main():
    model = YOLO(
        "runs/detect/road_damage_extended/weights/last.pt"
    )
    model.train(resume=True)

if __name__ == "__main__":
    main()