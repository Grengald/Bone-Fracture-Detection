import streamlit as st
import requests
import base64
from PIL import Image
import io

st.title("Bone Fracture Detection")

uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Original Image")

    if st.button("Analyze"):

        files = {"file": uploaded_file.getvalue()}

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files=files
        )

        if response.status_code == 200:

            data = response.json()

            fracture = data["fracture"]
            confidence = data["confidence"]

            st.subheader("Result")

            if fracture:
                st.error(f"Fracture detected (confidence: {confidence:.2f})")
            else:
                st.success(f"No fracture detected (confidence: {confidence:.2f})")

            # Декодируем изображение
            img_bytes = base64.b64decode(data["image"])
            img = Image.open(io.BytesIO(img_bytes))

            st.image(img, caption="Detection Result")