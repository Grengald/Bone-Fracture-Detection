from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from pathlib import Path
import gdown

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "best.pt"

GDRIVE_URL = "https://drive.google.com/uc?id=1jAPQ-Id_ZDvmFuI6F9jn1ZGHa8wUQbOC"

CONF_THRESHOLD = 0.5


def load_model():
    if not WEIGHTS_PATH.exists():
        print("⬇️ Downloading YOLO weights from Google Drive...")
        gdown.download(GDRIVE_URL, str(WEIGHTS_PATH), quiet=False)

    print("🚀 Loading YOLO model...")
    return YOLO(str(WEIGHTS_PATH))


model = load_model()


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
        fracture = confidence >= CONF_THRESHOLD

    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "fracture": fracture,
        "confidence": confidence,
        "image": img_base64
    })
