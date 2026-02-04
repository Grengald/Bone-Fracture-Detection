import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Bone Fracture Detection", layout="wide")

st.title("🦴 Bone Fracture Detection")

CONF_THRESHOLD = 0.5

# Загружаем модель один раз
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # <- твои веса

model = load_model()

uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Analyze"):
        # Конвертируем в формат для YOLO
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Детекция
        results = model(img)
        boxes = results[0].boxes

        fracture = False
        confidence = 0.0

        if boxes is not None and len(boxes) > 0:
            confidence = float(boxes.conf.max())
            fracture = confidence >= CONF_THRESHOLD

        # Аннотированное изображение
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Результат
        st.subheader("Result")
        if fracture:
            st.error(f"Fracture detected (confidence: {confidence:.2f})")
        else:
            st.success(f"No fracture detected (confidence: {confidence:.2f})")

        st.image(annotated, caption="Detection Result", use_container_width=True)
