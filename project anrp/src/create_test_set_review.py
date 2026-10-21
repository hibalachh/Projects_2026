
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = ROOT / "outputs" / "anpr" / "paddleocr_anpr_results.csv"
GROUND_TRUTH_FILE = ROOT / "outputs" / "ocr_preprocessing" / "ocr_ground_truth.csv"
OUTPUT_FILE = ROOT / "outputs" / "evaluation" / "test_set_ground_truth_review.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

predictions = pd.read_csv(PREDICTIONS_FILE)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

# Normalize crop filenames so predictions can be matched to existing labels.
predictions["crop_name"] = predictions["crop_path"].fillna("").apply(
    lambda p: Path(str(p)).name if str(p).strip() else ""
)
ground_truth["crop_name"] = ground_truth["image"].fillna("").apply(
    lambda p: Path(str(p)).name
)

# Keep known nonblank ground-truth labels.
known_labels = ground_truth[
    ground_truth["ground_truth"].notna()
    & ground_truth["ground_truth"].astype(str).str.strip().ne("")
].drop_duplicates("crop_name")

label_lookup = dict(zip(known_labels["crop_name"], known_labels["ground_truth"]))

rows = []

for image_name, group in predictions.groupby("image", sort=True):
    detected_crops = []

    for _, row in group.iterrows():
        crop_name = row["crop_name"]
        plate_text = (
            str(row["plate_number"])
            if pd.notna(row["plate_number"]) and str(row["plate_number"]).strip()
            else "[NO TEXT]"
        )

        if crop_name:
            detected_crops.append(
                f"{crop_name}: {plate_text} "
                f"(YOLO={row['detection_confidence']})"
            )
        else:
            detected_crops.append("No plate detected")

    # Prefill a known label if one exists for this image.
    known_for_image = [
        label_lookup[crop]
        for crop in group["crop_name"]
        if crop in label_lookup
    ]

    existing_truth = known_for_image[0] if known_for_image else ""

    rows.append({
        "image": image_name,
        "plate_visible": "",
        "ground_truth_plate": existing_truth,
        "predicted_crops": " | ".join(detected_crops),
        "reviewed": ""
    })

review = pd.DataFrame(rows)
review.to_csv(OUTPUT_FILE, index=False)

print(f"Images prepared for review: {len(review)}")
print(f"Review file created:\n{OUTPUT_FILE}")
print("\nOpen the CSV and fill in:")
print("- plate_visible: yes or no")
print("- ground_truth_plate: actual plate text, or blank if no plate exists")
print("- reviewed: yes after checking the original image")
