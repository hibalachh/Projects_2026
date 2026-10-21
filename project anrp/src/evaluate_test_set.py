
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = ROOT / "outputs" / "anpr" / "paddleocr_anpr_results.csv"
GROUND_TRUTH_FILE = ROOT / "outputs" / "evaluation" / "test_set_ground_truth_review.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value):
    """Normalize plate text for comparison."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def character_error_rate(prediction, ground_truth):
    """Calculate edit distance divided by ground-truth length."""
    pred = normalize_text(prediction)
    truth = normalize_text(ground_truth)

    if not truth:
        return None

    previous = list(range(len(pred) + 1))

    for i, truth_char in enumerate(truth, start=1):
        current = [i]

        for j, pred_char in enumerate(pred, start=1):
            cost = 0 if truth_char == pred_char else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1] / len(truth)


# Load data
predictions = pd.read_csv(PREDICTIONS_FILE)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

required_pred = {
    "image", "plate_number", "detection_confidence", "crop_path"
}
required_gt = {
    "image", "plate_visible", "ground_truth_plate", "reviewed"
}

if not required_pred.issubset(predictions.columns):
    raise ValueError(
        f"Prediction CSV is missing columns: "
        f"{required_pred - set(predictions.columns)}"
    )

if not required_gt.issubset(ground_truth.columns):
    raise ValueError(
        f"Ground-truth CSV is missing columns: "
        f"{required_gt - set(ground_truth.columns)}"
    )


# Match using filename stems to handle paths consistently
predictions["image_key"] = predictions["image"].apply(
    lambda x: Path(str(x)).stem if pd.notna(x) else ""
)
ground_truth["image_key"] = ground_truth["image"].apply(
    lambda x: Path(str(x)).stem if pd.notna(x) else ""
)

# A non-empty crop path indicates a recorded plate crop
predictions["has_crop"] = (
    predictions["crop_path"].fillna("").astype(str).str.strip().ne("")
)

predictions["plate_number"] = predictions["plate_number"].fillna("").astype(str)
predictions["detection_confidence"] = pd.to_numeric(
    predictions["detection_confidence"], errors="coerce"
)

# Keep the highest-confidence crop for each image.
# This is a single-plate-per-image evaluation.
candidates = predictions[predictions["has_crop"]].copy()
candidates = candidates.sort_values(
    "detection_confidence", ascending=False, na_position="last"
)
best_prediction = candidates.drop_duplicates(
    subset="image_key", keep="first"
)

best_prediction = best_prediction[
    ["image_key", "plate_number", "detection_confidence", "crop_path"]
].rename(
    columns={
        "plate_number": "predicted_plate",
        "detection_confidence": "best_detection_confidence",
        "crop_path": "selected_crop_path",
    }
)

# Create one row per reviewed source image
results = ground_truth.merge(
    best_prediction,
    on="image_key",
    how="left",
    validate="one_to_one",
)

results["predicted_plate"] = results["predicted_plate"].fillna("")
results["plate_visible"] = (
    results["plate_visible"].fillna("").astype(str).str.strip().str.lower()
)
results["reviewed"] = (
    results["reviewed"].fillna("").astype(str).str.strip().str.lower()
)

results["detection_found"] = results["selected_crop_path"].notna()

results["ground_truth_normalized"] = results["ground_truth_plate"].apply(
    normalize_text
)
results["prediction_normalized"] = results["predicted_plate"].apply(
    normalize_text
)

# Only compare OCR when a visible plate has a reviewed, non-empty label
results["eligible_for_ocr"] = (
    results["plate_visible"].eq("yes")
    & results["reviewed"].eq("yes")
    & results["ground_truth_normalized"].ne("")
)

results["exact_match"] = pd.NA
results["cer"] = pd.NA

eligible = results["eligible_for_ocr"]

results.loc[eligible, "exact_match"] = (
    results.loc[eligible, "ground_truth_normalized"]
    == results.loc[eligible, "prediction_normalized"]
)

results.loc[eligible, "cer"] = results.loc[eligible].apply(
    lambda row: character_error_rate(
        row["predicted_plate"], row["ground_truth_plate"]
    ),
    axis=1,
)

# Identify cases requiring more work
results["evaluation_note"] = ""

results.loc[
    results["plate_visible"].eq("yes")
    & results["ground_truth_normalized"].eq(""),
    "evaluation_note",
] = "Ground-truth plate text needed"

results.loc[
    results["plate_visible"].eq("yes")
    & ~results["detection_found"],
    "evaluation_note",
] = "Visible plate but no recorded detection"

results.loc[
    results["plate_visible"].eq("no"),
    "evaluation_note",
] = "Marked as no visible plate"

# Save detailed image-level results
detail_file = OUTPUT_DIR / "full_test_set_evaluation.csv"
results.to_csv(detail_file, index=False)

# Calculate metrics
visible = results["plate_visible"].eq("yes")
not_visible = results["plate_visible"].eq("no")
ocr_results = results[results["eligible_for_ocr"]].copy()

visible_count = int(visible.sum())
detected_visible = int(
    (visible & results["detection_found"]).sum()
)
labeled_count = len(ocr_results)
exact_count = int(ocr_results["exact_match"].fillna(False).sum())
cer_values = pd.to_numeric(ocr_results["cer"], errors="coerce").dropna()

summary = [
    ("Total test images", len(results)),
    ("Images marked plate visible", visible_count),
    ("Images marked no visible plate", int(not_visible.sum())),
    ("Visible-plate images with a recorded detection", detected_visible),
    (
        "Detection coverage on visible-plate images (%)",
        round(detected_visible / visible_count * 100, 2)
        if visible_count else None,
    ),
    ("Images eligible for OCR evaluation", labeled_count),
    ("OCR exact matches", exact_count),
    (
        "OCR exact-match accuracy (%)",
        round(exact_count / labeled_count * 100, 2)
        if labeled_count else None,
    ),
    (
        "Mean character error rate (%)",
        round(cer_values.mean() * 100, 2)
        if len(cer_values) else None,
    ),
    (
        "Visible images needing ground-truth text",
        int((visible & results["ground_truth_normalized"].eq("")).sum()),
    ),
]

summary_df = pd.DataFrame(summary, columns=["metric", "value"])
summary_file = OUTPUT_DIR / "full_test_set_summary.csv"
summary_df.to_csv(summary_file, index=False)

print("\n=== FULL TEST-SET ANPR EVALUATION ===")
for metric, value in summary:
    print(f"{metric}: {value}")

print(f"\nPer-image results saved to:\n{detail_file}")
print(f"Summary saved to:\n{summary_file}")

print("\n=== OCR RESULTS FOR LABELED IMAGES ===")
display_columns = [
    "image",
    "ground_truth_plate",
    "predicted_plate",
    "detection_found",
    "exact_match",
    "cer",
]
print(ocr_results[display_columns].to_string(index=False))

print("\n=== ITEMS NEEDING ATTENTION ===")
attention = results[results["evaluation_note"].ne("")]
print(
    attention[["image", "evaluation_note"]].to_string(index=False)
    if not attention.empty
    else "None"
)
