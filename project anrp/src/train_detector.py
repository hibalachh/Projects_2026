
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "processed" / "license_plates" / "dataset.yaml"
RUNS_DIR = ROOT / "outputs" / "runs"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print("Starting license-plate detector training on CPU...")
    model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        patience=5,
        seed=42,
        project=str(RUNS_DIR),
        name="license_plate_detector",
        plots=True,
    )

    print("\nTraining finished.")
    print(f"Expected best model: {RUNS_DIR / 'license_plate_detector' / 'weights' / 'best.pt'}")
    print(f"Expected last model: {RUNS_DIR / 'license_plate_detector' / 'weights' / 'last.pt'}")

if __name__ == "__main__":
    main()
