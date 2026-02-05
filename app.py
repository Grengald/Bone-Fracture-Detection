import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import gdown

# Загрузка модели
@st.cache_resource
def load_model():
    weights_path = "best.pt"
    gdrive_url = "https://drive.google.com/uc?id=1jAPQ-Id_ZDvmFuI6F9jn1ZGHa8wUQbOC"
    
    if not os.path.exists(weights_path):
        with st.spinner("Скачиваем модель..."):
            gdown.download(gdrive_url, weights_path, quiet=False)
    
    return YOLO(weights_path)

# Интерфейс
st.set_page_config(page_title="Детектор переломов", layout="wide")

st.title("🦴 Детектор переломов костей")
st.markdown("Загрузите рентгеновский снимок")

uploaded_file = st.file_uploader("Выберите изображение", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Загружаем модель
    model = load_model()
    
    # Показываем оригинал
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Исходное изображение")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("📊 Результаты")
        
        if st.button("🔍 Анализировать", type="primary"):
            with st.spinner("Анализируем..."):
                # Конвертируем в OpenCV формат
                image_np = np.array(image)
                if len(image_np.shape) == 2:  # Если черно-белое
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                else:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Детекция
                results = model(image_np)
                boxes = results[0].boxes
                
                # Результаты
                fracture = False
                confidence = 0.0
                
                if boxes is not None and len(boxes) > 0:
                    confidence = float(boxes.conf.max())
                    fracture = confidence >= 0.5
                
                # Показываем результат
                if fracture:
                    st.error(f"⚠️ Обнаружен перелом!")
                else:
                    st.success("✅ Переломов не обнаружено")
                
                st.metric("Уверенность модели", f"{confidence:.2%}")
                
                # Показываем аннотированное изображение
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Результат детекции", use_column_width=True)
