
from pathlib import Path
import re

import pandas as pd


GROUND_TRUTH_PATH = Path(
    "outputs/evaluation/test_set_ground_truth_review.csv"
)
PREDICTIONS_PATH = Path(
    "outputs/anpr/paddleocr_anpr_results.csv"
)
OUTPUT_DIR = Path("outputs/evaluation")

# Conservative threshold: flag low-confidence detections for review.
MIN_DETECTION_CONFIDENCE = 0.30
MIN_OCR_CONFIDENCE = 0.50


def normalize_text(value):
    """Normalize text for comparison, without guessing characters."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def main():
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    pred = pd.read_csv(PREDICTIONS_PATH)

    # Match image names whether they include .png or not.
    gt["image_key"] = gt["image"].astype(str).apply(
        lambda x: Path(x).stem
    )
    pred["image_key"] = pred["image"].astype(str).apply(
        lambda x: Path(x).stem
    )

    # Keep all raw detections in the original CSV.
    # For final prediction, choose the candidate with the highest
    # YOLO detection confidence among candidates with OCR text.
    pred["plate_number"] = pred["plate_number"].fillna("").astype(str)
    pred["detection_confidence"] = pd.to_numeric(
        pred["detection_confidence"], errors="coerce"
    )
    pred["ocr_confidence"] = pd.to_numeric(
        pred["ocr_confidence"], errors="coerce"
    )

    candidates = pred[
        (pred["plate_number"].str.strip() != "")
        & (pred["detection_confidence"].notna())
    ].copy()

    candidates = candidates.sort_values(
        "detection_confidence", ascending=False
    )

    best = candidates.drop_duplicates("image_key", keep="first")

    result = gt.merge(
        best[
            [
                "image_key",
                "plate_number",
                "detection_confidence",
                "ocr_confidence",
                "crop_path",
                "status",
            ]
        ],
        on="image_key",
        how="left",
    )

    result = result.rename(
        columns={"plate_number": "predicted_plate"}
    )

    result["ground_truth_normalized"] = (
        result["ground_truth_plate"].apply(normalize_text)
    )
    result["prediction_normalized"] = (
        result["predicted_plate"].apply(normalize_text)
    )

    result["eligible_for_ocr_evaluation"] = (
        result["plate_visible"].astype(str).str.lower().eq("yes")
        & result["reviewed"].astype(str).str.lower().eq("yes")
        & result["ground_truth_normalized"].ne("")
    )

    result["exact_match"] = pd.NA
    eligible = result["eligible_for_ocr_evaluation"]

    result.loc[eligible, "exact_match"] = (
        result.loc[eligible, "ground_truth_normalized"]
        == result.loc[eligible, "prediction_normalized"]
    )

    result["needs_review"] = (
        result["predicted_plate"].isna()
        | result["detection_confidence"].fillna(0).lt(
            MIN_DETECTION_CONFIDENCE
        )
        | result["ocr_confidence"].fillna(0).lt(MIN_OCR_CONFIDENCE)
    )

    # Do not count missing labels as OCR errors.
    evaluated = result[result["eligible_for_ocr_evaluation"]]
    exact_matches = int(evaluated["exact_match"].eq(True).sum())
    evaluated_count = len(evaluated)

    summary = pd.DataFrame(
        [
            {
                "total_test_images": len(result),
                "visible_plate_images": int(
                    result["plate_visible"].astype(str)
                    .str.lower().eq("yes").sum()
                ),
                "labelled_images_evaluated": evaluated_count,
                "exact_matches": exact_matches,
                "exact_match_accuracy_percent": round(
                    100 * exact_matches / evaluated_count, 2
                ) if evaluated_count else 0,
                "predictions_flagged_for_review": int(
                    result["needs_review"].sum()
                ),
                "ground_truth_labels_missing": int(
                    (
                        result["plate_visible"].astype(str)
                        .str.lower().eq("yes")
                        & result["ground_truth_normalized"].eq("")
                    ).sum()
                ),
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUTPUT_DIR / "final_predictions_evaluation.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "final_predictions_summary.csv",
        index=False,
    )

    print("\n=== FINAL-PREDICTION BASELINE ===")
    print(summary.to_string(index=False))
    print("\nSaved detailed results to:")
    print(OUTPUT_DIR / "final_predictions_evaluation.csv")
    print("\nSaved summary to:")
    print(OUTPUT_DIR / "final_predictions_summary.csv")


if __name__ == "__main__":
    main()
