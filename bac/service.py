from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = FastAPI()

# ===== Загружаем веса =====
model = YOLO("best.pt")   # <-- вставь путь к своим весам

CONF_THRESHOLD = 0.5


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    boxes = results[0].boxes

    fracture = False
    confidence = 0.0

    if boxes is not None and len(boxes) > 0:
        confidence = float(boxes.conf.max())

        if confidence > CONF_THRESHOLD:
            fracture = True

    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "fracture": fracture,
        "confidence": confidence,
        "image": img_base64
    })
