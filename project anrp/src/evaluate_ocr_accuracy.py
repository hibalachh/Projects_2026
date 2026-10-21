
import pandas as pd
import re
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing" / "ocr_ground_truth.csv"
)
PREDICTIONS_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing"
    / "ocr_preprocessing_comparison.csv"
)
OUTPUT_FILE = (
    BASE_DIR / "outputs" / "ocr_preprocessing"
    / "ocr_accuracy_results.csv"
)


def normalize(text):
    """Uppercase plate text and remove spaces and punctuation."""
    if pd.isna(text):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def edit_distance(reference, prediction):
    """Calculate the number of character insertions, deletions and substitutions."""
    previous = list(range(len(prediction) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, pred_char in enumerate(prediction, start=1):
            cost = 0 if ref_char == pred_char else 1
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1]


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Cannot find the {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def main():
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(f"Missing file: {GROUND_TRUTH_FILE}")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PREDICTIONS_FILE}")

    truth = pd.read_csv(GROUND_TRUTH_FILE, encoding="utf-8-sig")
    predictions = pd.read_csv(PREDICTIONS_FILE, encoding="utf-8-sig")

    # Identify columns in the existing comparison file.
    image_col = find_column(
        predictions,
        ["image", "image_name", "filename", "crop", "crop_name"],
        "image filename",
    )
    method_col = find_column(
        predictions,
        ["method", "preprocessing", "preprocessing_method"],
        "preprocessing method",
    )
    text_col = find_column(
        predictions,
        ["predicted_text", "ocr_text", "recognized_text",
         "detected_text", "text", "result"],
        "OCR prediction",
    )

    # Ignore images without verified ground truth.
    truth["ground_truth"] = truth["ground_truth"].fillna("").astype(str).str.strip()
    truth = truth[truth["ground_truth"] != ""].copy()
    truth["expected"] = truth["ground_truth"].apply(normalize)

    predictions = predictions.merge(
        truth[["image", "expected"]],
        left_on=image_col,
        right_on="image",
        how="inner",
    )

    if predictions.empty:
        raise ValueError(
            "No matching images found. Check the image filenames "
            "in both CSV files."
        )

    predictions["predicted"] = predictions[text_col].apply(normalize)
    predictions["exact_match"] = (
        predictions["predicted"] == predictions["expected"]
    )
    predictions["edit_distance"] = predictions.apply(
        lambda row: edit_distance(row["expected"], row["predicted"]),
        axis=1,
    )
    predictions["character_error_rate"] = predictions.apply(
        lambda row: (
            row["edit_distance"] / len(row["expected"])
            if row["expected"] else float("nan")
        ),
        axis=1,
    )

    # Save the individual comparisons.
    predictions.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Summarize each preprocessing method.
    summary = predictions.groupby(method_col).agg(
        images_evaluated=("exact_match", "count"),
        exact_matches=("exact_match", "sum"),
        exact_match_accuracy=("exact_match", "mean"),
        mean_character_error_rate=("character_error_rate", "mean"),
    )

    summary["exact_match_accuracy"] *= 100
    summary["mean_character_error_rate"] *= 100

    print("\nOCR ACCURACY BY PREPROCESSING METHOD")
    print(summary.round(2).to_string())

    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")
    print(
        f"\nVerified images evaluated: "
        f"{truth['image'].nunique()}"
    )
    print(
        "Note: These results only measure the images with "
        "non-empty verified ground truth."
    )


if __name__ == "__main__":
    main()
