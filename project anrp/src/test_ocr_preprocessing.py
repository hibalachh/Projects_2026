
from pathlib import Path
import cv2
import easyocr

ROOT = Path(__file__).resolve().parent.parent
CROP_DIR = ROOT / "outputs" / "anpr" / "plate_crops"
OUTPUT_DIR = ROOT / "outputs" / "ocr_preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with a plate crop that OCR struggled to read.
crop_path = CROP_DIR / "Cars15_plate_1.jpg"

if not crop_path.exists():
    raise FileNotFoundError(f"Crop not found: {crop_path}")

image = cv2.imread(str(crop_path))
if image is None:
    raise ValueError(f"Could not open image: {crop_path}")

# Enlarge the crop to help OCR distinguish characters.
scale = 3
enlarged = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC,
)

# Convert to grayscale and improve local contrast.
gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Create a second version with a mild denoising step.
denoised = cv2.fastNlMeansDenoising(
    enhanced,
    None,
    h=10,
    templateWindowSize=7,
    searchWindowSize=21,
)

cv2.imwrite(str(OUTPUT_DIR / "original_enlarged.jpg"), enlarged)
cv2.imwrite(str(OUTPUT_DIR / "grayscale_enhanced.jpg"), enhanced)
cv2.imwrite(str(OUTPUT_DIR / "denoised.jpg"), denoised)

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

for name, image_variant in [
    ("Original crop", image),
    ("Enlarged", enlarged),
    ("Contrast enhanced", enhanced),
    ("Enhanced and denoised", denoised),
]:
    results = reader.readtext(
        image_variant,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    print(f"\n{name}:")
    if not results:
        print("  No text detected")
    for _, text, confidence in results:
        print(f"  Text: {text} | Confidence: {confidence:.4f}")

print(f"\nComparison images saved to: {OUTPUT_DIR}")
