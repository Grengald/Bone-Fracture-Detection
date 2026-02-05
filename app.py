# app.py
import os
os.environ["ULTRALYTICS_NO_GUI"] = "1"  # отключаем GUI OpenCV
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
import gdown

# ----------------- НАСТРОЙКИ -----------------
st.set_page_config(page_title="Bone Fracture Detection", layout="centered")
st.title("🦴 Bone Fracture Detection")

CONF_THRESHOLD = 0.5
WEIGHTS_PATH = Path("best.pt")
GDRIVE_URL = "https://drive.google.com/uc?id=1jAPQ-Id_ZDvmFuI6F9jn1ZGHa8wUQbOC"

# ----------------- ФУНКЦИИ -----------------
@st.cache_resource
def download_model():
    if not WEIGHTS_PATH.exists():
        st.info("⬇️ Downloading YOLO weights from Google Drive...")
        gdown.download(GDRIVE_URL, str(WEIGHTS_PATH), quiet=False)
    st.success("✅ YOLO weights ready")
    return YOLO(str(WEIGHTS_PATH))

@st.cache_resource
def load_model():
    return download_model()

def predict_fracture(img):
    results = model(img)
    boxes = results[0].boxes

    fracture = False
    confidence = 0.0

    if boxes is not None and len(boxes) > 0:
        # Получаем максимальную уверенность по всем боксам
        max_conf = float(boxes.conf.max())
        confidence = max_conf
        fracture = max_conf >= CONF_THRESHOLD

        # Фильтруем боксы ниже порога
        filtered_boxes = [b for b in boxes if float(b.conf[0]) >= CONF_THRESHOLD]
        if filtered_boxes:
            results[0].boxes = type(boxes)(filtered_boxes)
        else:
            # Если нет боксов выше порога, оставляем пустой список
            results[0].boxes = type(boxes)([])
    else:
        results[0].boxes = type(boxes)([])  # если боксов нет, пустой список

    annotated = results[0].plot()
    return fracture, confidence, annotated

# ----------------- ЗАГРУЗКА МОДЕЛИ -----------------
model = load_model()

# ----------------- ЗАГРУЗКА ИЗОБРАЖЕНИЯ -----------------
uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Конвертируем в OpenCV формат
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img[..., ::-1], caption="Original Image", use_column_width=True)

    if st.button("Analyze"):
        fracture, confidence, annotated = predict_fracture(img)

        st.subheader("Result")
        if fracture:
            st.error(f"Fracture detected (confidence: {confidence:.2f})")
        else:
            st.success(f"No fracture detected (confidence: {confidence:.2f})")

        st.image(annotated[..., ::-1], caption="Detection Result", use_column_width=True)
