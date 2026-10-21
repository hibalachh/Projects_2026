import re
import time
from pathlib import Path

import cv2
import easyocr
import pandas as pd
import torch  # Import before PaddleOCR
from paddleocr import PaddleOCR

BASE_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing"
    / "ocr_ground_truth.csv"
)
CROPS_DIR = BASE_DIR / "outputs" / "anpr" / "plate_crops"
OUTPUT_DIR = BASE_DIR / "outputs" / "ocr_preprocessing"

OUTPUT_FILE = OUTPUT_DIR / "easyocr_vs_paddleocr.csv"

DUPLICATE_TO_EXCLUDE = "Cars145_plate_1.jpg"


def normalize(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))

    for i, char_ref in enumerate(reference, start=1):
        current = [i]

        for j, char_pred in enumerate(prediction, start=1):
            cost = 0 if char_ref == char_pred else 1
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            ))

        previous = current

    return previous[-1]


def extract_paddle_text(result):
    """Extract recognized text from PaddleOCR 3.x output."""
    texts = []

    for item in result:
        data = getattr(item, "json", None)

        if callable(data):
            data = data()

        if data is None and isinstance(item, dict):
            data = item

        if isinstance(data, dict):
            data = data.get("res", data)
            recognized = data.get("rec_texts", [])

            if recognized:
                texts.extend(str(text) for text in recognized)
                continue

        # Fallback for a result format that exposes text directly.
        if isinstance(item, dict):
            recognized = item.get("rec_texts", [])
            texts.extend(str(text) for text in recognized)

    return "".join(texts)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(
        GROUND_TRUTH_FILE,
        encoding="utf-8-sig",
    )

    truth["ground_truth"] = (
        truth["ground_truth"].fillna("").astype(str).str.strip()
    )

    truth = truth[truth["ground_truth"] != ""].copy()
    truth = truth[
        truth["image"] != DUPLICATE_TO_EXCLUDE
    ].copy()

    print(f"Verified unique images: {len(truth)}")
    print("\nLoading EasyOCR...")
    easy_reader = easyocr.Reader(["en"], gpu=False)

    print("Loading PaddleOCR...")
    paddle_reader = PaddleOCR(
        lang="en",
        use_textline_orientation=False,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        )

    rows = []

    for _, row in truth.iterrows():
        image_name = row["image"]
        image_path = CROPS_DIR / image_name
        expected = normalize(row["ground_truth"])

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping missing/unreadable image: {image_name}")
            continue

        print(f"Processing {image_name}...")

        # EasyOCR
        start = time.perf_counter()
        easy_result = easy_reader.readtext(image, detail=1)
        easy_seconds = time.perf_counter() - start

        easy_text = "".join(item[1] for item in easy_result)

        # PaddleOCR 3.x
        start = time.perf_counter()
        paddle_result = paddle_reader.predict(
            input=str(image_path)
        )
        paddle_seconds = time.perf_counter() - start

        paddle_text = extract_paddle_text(paddle_result)

        for engine, raw_text, elapsed in [
            ("EasyOCR", easy_text, easy_seconds),
            ("PaddleOCR", paddle_text, paddle_seconds),
        ]:
            predicted = normalize(raw_text)
            errors = edit_distance(expected, predicted)

            rows.append({
                "image": image_name,
                "engine": engine,
                "expected": expected,
                "predicted": predicted,
                "exact_match": predicted == expected,
                "edit_distance": errors,
                "character_error_rate": (
                    errors / len(expected) if expected else None
                ),
                "processing_seconds": round(elapsed, 4),
            })

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if results.empty:
        print("No results generated. Check the image paths.")
        return

    summary = results.groupby("engine").agg(
        images_evaluated=("exact_match", "count"),
        exact_matches=("exact_match", "sum"),
        exact_match_accuracy=("exact_match", "mean"),
        mean_character_error_rate=("character_error_rate", "mean"),
        mean_processing_seconds=("processing_seconds", "mean"),
    )

    summary["exact_match_accuracy"] *= 100
    summary["mean_character_error_rate"] *= 100

    print("\nEASYOCR VS PADDLEOCR")
    print(summary.round(3).to_string())
    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
