
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing"
    / "ocr_accuracy_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing"
    / "ocr_accuracy_results_deduplicated.csv"
)

# Exclude one copy of the duplicated plate crop.
DUPLICATE_TO_EXCLUDE = "Cars145_plate_1.jpg"


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    original_images = df["image"].nunique()

    df = df[df["image"] != DUPLICATE_TO_EXCLUDE].copy()

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    summary = df.groupby("method").agg(
        images_evaluated=("exact_match", "count"),
        exact_matches=("exact_match", "sum"),
        exact_match_accuracy=("exact_match", "mean"),
        mean_character_error_rate=("character_error_rate", "mean"),
    )

    summary["exact_match_accuracy"] *= 100
    summary["mean_character_error_rate"] *= 100

    print("\nDEDUPLICATED OCR EVALUATION")
    print(summary.round(2).to_string())
    print(f"\nUnique images before exclusion: {original_images}")
    print(f"Excluded duplicate: {DUPLICATE_TO_EXCLUDE}")
    print(f"\nResults saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
