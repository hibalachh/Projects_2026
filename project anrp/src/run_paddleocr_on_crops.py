
import json
from pathlib import Path

import pandas as pd
import torch
from paddleocr import PaddleOCR


BASE_DIR = Path(__file__).resolve().parent.parent

CROPS_DIR = BASE_DIR / "outputs" / "anpr" / "plate_crops"
OUTPUT_DIR = BASE_DIR / "outputs" / "anpr"
OUTPUT_FILE = OUTPUT_DIR / "paddleocr_results.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_result_data(result_item):
    """Convert a PaddleOCR result object to a dictionary."""
    data = getattr(result_item, "json", None)

    if callable(data):
        data = data()

    if isinstance(data, str):
        data = json.loads(data)

    if isinstance(data, dict):
        return data.get("res", data)

    return {}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CROPS_DIR.exists():
        raise FileNotFoundError(
            f"Plate crops directory not found: {CROPS_DIR}"
        )

    image_paths = sorted(
        path for path in CROPS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No plate crops found in {CROPS_DIR}")
        return

    print(f"Plate crops found: {len(image_paths)}")
    print("Loading PaddleOCR...")

    reader = PaddleOCR(
        lang="en",
        use_textline_orientation=False,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    rows = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")

        try:
            results = reader.predict(input=str(image_path))

            recognized_text = []
            confidence_scores = []

            for item in results:
                data = get_result_data(item)

                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])

                recognized_text.extend(str(text) for text in texts)
                confidence_scores.extend(
                    float(score) for score in scores
                )

            text = "".join(recognized_text)
            mean_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores else None
            )

            rows.append({
                "image": image_path.name,
                "recognized_text": text,
                "mean_confidence": mean_confidence,
                "status": "text_read" if text.strip() else "ocr_no_text",
                "error": "",
            })

        except Exception as exc:
            rows.append({
                "image": image_path.name,
                "recognized_text": "",
                "mean_confidence": None,
                "status": "error",
                "error": str(exc),
            })
            print(f"  OCR failed: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nPADDLEOCR CROP RESULTS")
    print(f"Total crops: {len(df)}")
    print(f"Text recognized: {(df['status'] == 'text_read').sum()}")
    print(f"No text: {(df['status'] == 'ocr_no_text').sum()}")
    print(f"Errors: {(df['status'] == 'error').sum()}")
    print(f"\nResults saved to:\n{OUTPUT_FILE}")

    print("\nFirst 10 results:")
    print(df[["image", "recognized_text", "status"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
