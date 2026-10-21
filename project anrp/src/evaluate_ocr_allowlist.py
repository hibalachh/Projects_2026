
import re
import cv2
import easyocr
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing"
    / "ocr_ground_truth.csv"
)
CROPS_DIR = BASE_DIR / "outputs" / "anpr" / "plate_crops"
OUTPUT_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing"
    / "ocr_allowlist_results.csv"
)

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


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


def main():
    truth = pd.read_csv(
        GROUND_TRUTH_FILE, encoding="utf-8-sig"
    )
    truth["ground_truth"] = (
        truth["ground_truth"].fillna("").astype(str).str.strip()
    )
    truth = truth[truth["ground_truth"] != ""].copy()

    # Exclude the duplicate crop already identified.
    truth = truth[
        truth["image"] != "Cars145_plate_1.jpg"
    ].copy()

    reader = easyocr.Reader(["en"], gpu=False)
    results = []

    for _, row in truth.iterrows():
        image_name = row["image"]
        image_path = CROPS_DIR / image_name

        if not image_path.exists():
            print(f"Missing image: {image_name}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not open: {image_name}")
            continue

        expected = normalize(row["ground_truth"])

        # Experiment A: standard EasyOCR.
        standard = reader.readtext(image, detail=1)
        standard_text = "".join(item[1] for item in standard)
        standard_conf = (
            sum(item[2] for item in standard) / len(standard)
            if standard else 0.0
        )

        # Experiment B: EasyOCR restricted to plate characters.
        restricted = reader.readtext(
            image,
            detail=1,
            allowlist=ALLOWLIST,
        )
        restricted_text = "".join(item[1] for item in restricted)
        restricted_conf = (
            sum(item[2] for item in restricted) / len(restricted)
            if restricted else 0.0
        )

        for method, text, confidence in [
            ("standard", standard_text, standard_conf),
            ("allowlist", restricted_text, restricted_conf),
        ]:
            predicted = normalize(text)
            errors = edit_distance(expected, predicted)

            results.append({
                "image": image_name,
                "method": method,
                "expected": expected,
                "predicted": predicted,
                "confidence": round(confidence, 4),
                "exact_match": predicted == expected,
                "edit_distance": errors,
                "character_error_rate": (
                    errors / len(expected) if expected else None
                ),
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    if df.empty:
        print("No results were produced. Check your crop paths.")
        return

    summary = df.groupby("method").agg(
        images_evaluated=("exact_match", "count"),
        exact_matches=("exact_match", "sum"),
        exact_match_accuracy=("exact_match", "mean"),
        mean_character_error_rate=("character_error_rate", "mean"),
    )

    summary["exact_match_accuracy"] *= 100
    summary["mean_character_error_rate"] *= 100

    print("\nOCR ALLOWLIST EXPERIMENT")
    print(summary.round(2).to_string())
    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
