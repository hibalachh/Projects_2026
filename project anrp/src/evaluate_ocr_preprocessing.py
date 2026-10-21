
from pathlib import Path
import csv
import re

import cv2
import easyocr

ROOT = Path(__file__).resolve().parent.parent
CROP_DIR = ROOT / "outputs" / "anpr" / "plate_crops"
OUTPUT_DIR = ROOT / "outputs" / "ocr_preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not CROP_DIR.exists():
    raise FileNotFoundError(f"Plate crop directory not found: {CROP_DIR}")

# Select up to 15 crops for a first comparison.
crop_paths = sorted(CROP_DIR.glob("*.jpg"))[:15]

if not crop_paths:
    raise FileNotFoundError(f"No JPG crops found in: {CROP_DIR}")

print("Loading EasyOCR on CPU...")
reader = easyocr.Reader(["en"], gpu=False)

def normalize_text(text):
    """Remove spaces and punctuation for easier comparison."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def preprocess_variants(image):
    """Return the original and three processed versions."""
    enlarged = cv2.resize(
        image, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return {
        "original": image,
        "enlarged": enlarged,
        "contrast_enhanced": enhanced,
        "denoised": denoised,
    }

rows = []

for index, crop_path in enumerate(crop_paths, start=1):
    image = cv2.imread(str(crop_path))

    if image is None:
        print(f"Skipping unreadable image: {crop_path.name}")
        continue

    variants = preprocess_variants(image)
    print(f"[{index}/{len(crop_paths)}] {crop_path.name}")

    for method, variant in variants.items():
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )

        # Combine detected text fragments into one string.
        text = "".join(item[1] for item in results)
        confidences = [float(item[2]) for item in results]
        mean_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        rows.append({
            "image": crop_path.name,
            "method": method,
            "recognized_text": text,
            "normalized_text": normalize_text(text),
            "mean_ocr_confidence": round(mean_confidence, 4),
            "text_detected": bool(text),
        })

csv_path = OUTPUT_DIR / "ocr_preprocessing_comparison.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "image",
        "method",
        "recognized_text",
        "normalized_text",
        "mean_ocr_confidence",
        "text_detected",
    ])
    writer.writeheader()
    writer.writerows(rows)

print("\n=== OCR PREPROCESSING SUMMARY ===")
print(f"Crops evaluated: {len(crop_paths)}")
print(f"Experiments recorded: {len(rows)}")
print(f"CSV results: {csv_path}")
print("\nCompare recognized text and confidence for each method.")
print("High confidence alone does not guarantee correct text.")
