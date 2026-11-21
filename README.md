# Road Damage Detection System using YOLO11n

A Deep Learning-based road damage detection application that identifies potholes and road cracks using YOLO11n and provides an interactive Streamlit interface.

## Features
- Detects five categories of road damage.
- Upload road images for detection.
- Displays bounding boxes and confidence scores.
- Adjustable confidence threshold.
- Damage detection summary and detailed results.
- Download annotated images and CSV inspection reports.

## Technologies
- Python
- PyTorch
- Ultralytics YOLO11n
- OpenCV
- Streamlit
- Pandas

## Dataset
**RDD2022 — Road Damage Detection Dataset**

Five classes:
1. Longitudinal crack
2. Transverse crack
3. Alligator crack
4. Other corruption
5. Pothole

The dataset contains 26,869 training images, 5,758 validation images and 5,758 test images.

Dataset source: https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022

## Model Training
- Architecture: YOLO11n
- Training epochs: 10
- Image size: 640 × 640
- Batch size: 8
- GPU: NVIDIA RTX 4060 Laptop GPU
- Pretrained YOLO11n weights fine-tuned on RDD2022.

## Test Results

| Metric | Score |
|---|---|
| Precision | 54.9% |
| Recall | 47.1% |
| mAP@50 | 47.9% |
| mAP@50–95 | 24.4% |

Evaluated on 5,758 test images.

## Running the Application

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the trained checkpoint at:

`runs/detect/road_damage_final/weights/best.pt`

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Limitations
- Performance varies across road conditions and lighting.
- False positives and missed damage can occur.
- Pothole detection requires further improvement.
- Results should be verified before making road maintenance decisions.

## Acknowledgements
This project uses the RDD2022 dataset, Ultralytics YOLO11n pretrained weights, and AI-assisted development support. The model was fine-tuned and evaluated locally.