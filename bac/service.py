from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pathlib import Path
import requests
import cv2
import numpy as np
import base64


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "best.pt"

WEIGHTS_URL = (
    "https://drive.google.com/uc?"
    "id=1jAPQ-Id_ZDvmFuI6F9jn1ZGHa8wUQbOC&export=download"
)

CONF_THRESHOLD = 0.5

# =====================
# DOWNLOAD WEIGHTS
# =====================

def download_weights():
    print("⬇️ Downloading model weights from Google Drive...")
    r = requests.get(WEIGHTS_URL, timeout=180)
    r.raise_for_status()

    WEIGHTS_PATH.write_bytes(r.content)

    size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Weights downloaded ({size_mb:.2f} MB)")

    if size_mb < 1:
        raise RuntimeError("Downloaded file is too small — probably not a .pt file")

if not WEIGHTS_PATH.exists():
    download_weights()



print("🚀 Loading YOLO model...")
model = YOLO(str(WEIGHTS_PATH))
print("✅ Model loaded")


app = FastAPI(title="Bone Fracture Detection API")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        results = model(img)[0]

        fracture = False
        confidence = 0.0

        for box in results.boxes:
            score = float(box.conf[0])
            if score > CONF_THRESHOLD:
                fracture = True
                confidence = max(confidence, score)

        annotated = results.plot()
        _, buffer = cv2.imencode(".jpg", annotated)

        return {
            "fracture": fracture,
            "confidence": round(confidence, 4),
            "image": base64.b64encode(buffer).decode()
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
