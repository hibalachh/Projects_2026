
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "processed" / "license_plates" / "dataset.yaml"
MODEL_PATH = (
    ROOT / "outputs" / "runs" / "license_plate_detector"
    / "weights" / "best.pt"
)
RUNS_DIR = ROOT / "outputs" / "runs"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    model = YOLO(str(MODEL_PATH))

    print("\n=== EVALUATING ON HELD-OUT TEST SET ===")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(RUNS_DIR),
        name="license_plate_test",
    )

    print("\n=== TEST METRICS ===")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    print("\nEvaluation outputs:")
    print(RUNS_DIR / "license_plate_test")


if __name__ == "__main__":
    main()
