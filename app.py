
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from io import BytesIO
import pandas as pd

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="centered"
)

st.title("Road Damage Detection System")
st.write("AI-powered road damage inspection using YOLO11n")

# Load our trained model
#@st.cache_resource
#def load_model():
@st.cache_resource
def load_model():
    return YOLO("runs/detect/road_damage_extended/weights/best.pt")
    #return YOLO("runs/detect/road_damage_final/weights/best.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

# Confidence threshold
confidence = st.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Run YOLO prediction
    results = model.predict(
        image,
        conf=confidence,
        imgsz=640
    )

    result = results[0]

    # Display annotated image
    annotated_image = Image.fromarray(
        result.plot()[:, :, ::-1]
    )

    st.image(
        annotated_image,
        caption="Detected Road Damage",
        use_container_width=True
    )

    # Extract detections
    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        score = float(box.conf.item())

        detections.append({
            "Damage Type": class_name,
            "Confidence": round(score, 3)
        })

    # Damage summary
    counts = Counter(
        item["Damage Type"] for item in detections
    )

    st.subheader("Damage Detection Summary")

    st.metric("Total Detections", len(detections))

    if detections:
        for damage_type, count in counts.items():
            st.write(f"{damage_type}: {count}")

        # Detection table
        st.subheader("Detection Details")
        df = pd.DataFrame(detections)
        st.dataframe(df, use_container_width=True)

    else:
        st.info("No road damage detected at this confidence threshold.")

    # Download annotated image
    buffer = BytesIO()
    annotated_image.save(buffer, format="PNG")

    st.download_button(
        label="Download Detected Image",
        data=buffer.getvalue(),
        file_name="road_damage_result.png",
        mime="image/png"
    )

    # Download CSV inspection report
    report = pd.DataFrame(
        detections,
        columns=["Damage Type", "Confidence"]
    )

    st.download_button(
        label="Download Inspection Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="road_damage_report.csv",
        mime="text/csv"
    )

st.caption(
    "Predictions are AI-generated and should be verified "
    "before making road maintenance decisions."
)
