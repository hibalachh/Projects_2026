
from pathlib import Path
import re

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = ROOT / "outputs" / "anpr" / "paddleocr_anpr_results.csv"
GROUND_TRUTH_FILE = ROOT / "outputs" / "ocr_preprocessing" / "ocr_ground_truth.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value):
    """Uppercase text and remove spaces and punctuation."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def character_error_rate(prediction, ground_truth):
    """Calculate character error rate using edit distance."""
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
                    current[-1] + 1,       # insertion
                    previous[j] + 1,       # deletion
                    previous[j - 1] + cost # substitution
                )
            )

        previous = current

    return previous[-1] / len(truth)


# Load files
predictions = pd.read_csv(PREDICTIONS_FILE)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

required_pred_cols = {"crop_path", "plate_number", "status"}
required_gt_cols = {"image", "ground_truth"}

if not required_pred_cols.issubset(predictions.columns):
    raise ValueError(
        f"Prediction CSV must contain: {required_pred_cols}"
    )

if not required_gt_cols.issubset(ground_truth.columns):
    raise ValueError(
        f"Ground-truth CSV must contain: {required_gt_cols}"
    )


# Match ground-truth crop filenames to prediction crop filenames
predictions["crop_name"] = predictions["crop_path"].fillna("").apply(
    lambda p: Path(str(p)).name if str(p).strip() else ""
)

ground_truth["crop_name"] = ground_truth["image"].fillna("").apply(
    lambda p: Path(str(p)).name
)

# Keep only nonblank, manually labeled ground-truth examples
labeled_gt = ground_truth[
    ground_truth["ground_truth"].notna()
    & ground_truth["ground_truth"].astype(str).str.strip().ne("")
].copy()

# Avoid accidentally counting duplicate ground-truth filenames twice
labeled_gt = labeled_gt.drop_duplicates(subset=["crop_name"], keep="first")

# Merge: each labeled crop is matched to its prediction
evaluation = labeled_gt.merge(
    predictions,
    on="crop_name",
    how="left",
    suffixes=("_gt", "_pred"),
    indicator=True
)

evaluation["plate_number"] = evaluation["plate_number"].fillna("")
evaluation["ground_truth_normalized"] = evaluation["ground_truth"].apply(
    normalize_text
)
evaluation["prediction_normalized"] = evaluation["plate_number"].apply(
    normalize_text
)

evaluation["exact_match"] = (
    evaluation["ground_truth_normalized"]
    == evaluation["prediction_normalized"]
)

evaluation["cer"] = evaluation.apply(
    lambda row: character_error_rate(
        row["plate_number"], row["ground_truth"]
    ),
    axis=1
)

evaluation["prediction_found"] = evaluation["_merge"].eq("both")

# Save per-crop comparisons
comparison_file = OUTPUT_DIR / "paddleocr_ground_truth_comparison.csv"
evaluation.to_csv(comparison_file, index=False)

# Metrics apply only to nonblank ground-truth crops in this file
total = len(evaluation)
exact = int(evaluation["exact_match"].sum())
cer_values = evaluation["cer"].dropna()

print("\n=== PADDLEOCR LABELED-CROP EVALUATION ===")
print(f"Ground-truth crops evaluated: {total}")
print(f"Predictions matched to crops: {int(evaluation['prediction_found'].sum())}")
print(f"Exact matches:                {exact}")
print(
    f"Exact-match accuracy:         "
    f"{exact / total * 100:.2f}%"
    if total else "Exact-match accuracy: N/A"
)
print(
    f"Mean character error rate:    "
    f"{cer_values.mean() * 100:.2f}%"
    if len(cer_values) else "Mean character error rate: N/A"
)
print(f"\nDetailed comparison saved to:\n{comparison_file}")

print("\n=== INDIVIDUAL RESULTS ===")
columns = [
    "crop_name",
    "ground_truth",
    "plate_number",
    "exact_match",
    "cer",
]
print(evaluation[columns].to_string(index=False))
