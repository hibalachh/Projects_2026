
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO
from paddleocr import PaddleOCR

from src.anpr_pipeline import (
    MODEL_PATH,
    IMAGE_SIZE,
    DETECTION_CONFIDENCE,
    expand_box,
    prepare_crop,
    extract_paddle_results,
)


detector = None
reader = None

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, reader

    if not MODEL_PATH.exists():
        raise RuntimeError(f"YOLO model not found: {MODEL_PATH}")

    print("Loading YOLO model...")
    detector = YOLO(str(MODEL_PATH))

    print("Loading PaddleOCR...")
    reader = PaddleOCR(
        lang="en",
        use_textline_orientation=False,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    print("ANPR API ready.")
    yield

    detector = None
    reader = None


app = FastAPI(
    title="ANPR Image Recognition API",
    description=(
        "Detect license plates in uploaded images and "
        "recognize their text using YOLO and PaddleOCR."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "project": "Automatic Number Plate Recognition",
        "mode": "images_only",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if detector is not None and reader is not None
        else "starting",
        "model_loaded": detector is not None,
        "ocr_loaded": reader is not None,
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, PNG, or BMP image.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB size limit.",
        )

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    height, width = image.shape[:2]

    prediction = detector.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=DETECTION_CONFIDENCE,
        device="cpu",
        verbose=False,
    )[0]

    plates = []

    if prediction.boxes is not None:
        for box in prediction.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            x1, y1, x2, y2 = expand_box(
                x1, y1, x2, y2, width, height
            )

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            plate_text = ""
            ocr_confidence = 0.0
            status = "ocr_no_text"

            try:
                ocr_image = prepare_crop(crop)
                ocr_results = reader.predict(input=ocr_image)
                plate_text, ocr_confidence = (
                    extract_paddle_results(ocr_results)
                )

                if plate_text:
                    status = "text_read"

            except Exception:
                status = "ocr_error"

            plates.append({
                "plate_text": plate_text,
                "detection_confidence": round(
                    float(box.conf[0].item()), 4
                ),
                "ocr_confidence": round(
                    float(ocr_confidence), 4
                ),
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "status": status,
            })

    plates.sort(
        key=lambda item: item["detection_confidence"],
        reverse=True,
    )

    return {
        "filename": Path(file.filename).name,
        "image_dimensions": {
            "width": width,
            "height": height,
        },
        "plate_count": len(plates),
        "plates": plates,
    }
