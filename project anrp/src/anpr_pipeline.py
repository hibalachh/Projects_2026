
from pathlib import Path
import csv
import re

import cv2
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR


# Project paths
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "outputs" / "runs" / "license_plate_detector"
    / "weights" / "best.pt"
)
IMAGE_DIR = (
    ROOT / "data" / "processed" / "license_plates"
    / "images" / "test"
)

OUTPUT_DIR = ROOT / "outputs" / "anpr"
CROPS_DIR = OUTPUT_DIR / "plate_crops"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"
CSV_PATH = OUTPUT_DIR / "paddleocr_anpr_results.csv"

# Detection settings
IMAGE_SIZE = 416
DETECTION_CONFIDENCE = 0.25


def expand_box(x1, y1, x2, y2, width, height, margin=0.08):
    """Add a small margin around a detected plate."""
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)

    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    )


def prepare_crop(crop):
    """Enlarge a plate crop to help OCR read small characters."""
    if crop.size == 0:
        return crop

    height, width = crop.shape[:2]

    return cv2.resize(
        crop,
        (max(1, width * 3), max(1, height * 3)),
        interpolation=cv2.INTER_CUBIC,
    )


def normalize_plate_text(text):
    """Keep only letters and digits in the recognized plate text."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def extract_paddle_results(results):
    """Extract recognized text and confidence scores from PaddleOCR 3.x."""
    recognized_text = []
    confidence_scores = []

    for item in results:
        data = getattr(item, "json", None)

        if callable(data):
            data = data()

        if isinstance(data, str):
            import json
            data = json.loads(data)

        if not isinstance(data, dict):
            continue

        data = data.get("res", data)

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])

        for text in texts:
            recognized_text.append(str(text))

        for score in scores:
            try:
                confidence_scores.append(float(score))
            except (TypeError, ValueError):
                pass

    combined_text = "".join(recognized_text)

    mean_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0.0
    )

    return normalize_plate_text(combined_text), mean_confidence


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained YOLO model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image folder not found: {IMAGE_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    print("Loading trained YOLO detector...")
    detector = YOLO(str(MODEL_PATH))

    print("Loading PaddleOCR on CPU...")
    print("Using the same OCR configuration as the successful comparison.")

    reader = PaddleOCR(
        lang="en",
        use_textline_orientation=False,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    rows = []
    total_plates = 0
    text_read_count = 0
    no_text_count = 0
    error_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        annotated = image.copy()

        result = detector.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=DETECTION_CONFIDENCE,
            device="cpu",
            verbose=False,
        )[0]

        detections = result.boxes

        print(f"[{index}/{len(image_paths)}] {image_path.name}")

        if detections is None or len(detections) == 0:
            rows.append({
                "image": image_path.name,
                "plate_number": "",
                "detection_confidence": "",
                "ocr_confidence": "",
                "crop_path": "",
                "status": "no_plate_detected",
            })

            cv2.imwrite(
                str(ANNOTATED_DIR / image_path.name),
                annotated,
            )
            continue

        for plate_index, box in enumerate(detections, start=1):
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            x1, y1, x2, y2 = expand_box(
                x1, y1, x2, y2, width, height
            )

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_name = (
                f"{image_path.stem}_plate_{plate_index}.jpg"
            )
            crop_path = CROPS_DIR / crop_name

            if not cv2.imwrite(str(crop_path), crop):
                print(f"Warning: could not save crop {crop_name}")
                continue

            # Preserve the original crop and enlarge only the OCR input.
            ocr_image = prepare_crop(crop)

            plate_text = ""
            ocr_confidence = 0.0
            status = "ocr_no_text"

            try:
                # PaddleOCR 3.x accepts an image path or image array.
                ocr_results = reader.predict(input=ocr_image)

                plate_text, ocr_confidence = extract_paddle_results(
                    ocr_results
                )

                if plate_text:
                    status = "text_read"
                    text_read_count += 1
                else:
                    no_text_count += 1

            except Exception as exc:
                status = "ocr_error"
                error_count += 1
                print(
                    f"  OCR error for {crop_name}: {exc}"
                )

            detection_confidence = float(box.conf[0].item())
            total_plates += 1

            rows.append({
                "image": image_path.name,
                "plate_number": plate_text,
                "detection_confidence": round(
                    detection_confidence, 4
                ),
                "ocr_confidence": round(ocr_confidence, 4),
                "crop_path": str(crop_path.relative_to(ROOT)),
                "status": status,
            })

            label = plate_text if plate_text else "Unreadable"
            label = f"{label} ({ocr_confidence:.2f})"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                label[:40],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(
            str(ANNOTATED_DIR / image_path.name),
            annotated,
        )

    fieldnames = [
        "image",
        "plate_number",
        "detection_confidence",
        "ocr_confidence",
        "crop_path",
        "status",
    ]

    with CSV_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== PADDLEOCR ANPR PIPELINE SUMMARY ===")
    print(f"Images in test set:       {len(image_paths)}")
    print(f"Plate crops processed:    {total_plates}")
    print(f"Text recognized:          {text_read_count}")
    print(f"No text recognized:       {no_text_count}")
    print(f"OCR errors:               {error_count}")
    print(f"CSV results:              {CSV_PATH}")
    print(f"Plate crops:              {CROPS_DIR}")
    print(f"Annotated images:         {ANNOTATED_DIR}")
    print("\nOCR results are preliminary; verify plate numbers against ground truth.")


if __name__ == "__main__":
    main()
