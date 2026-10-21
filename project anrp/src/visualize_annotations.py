
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "archive"
IMAGES = DATA / "images"
ANNOTATIONS = DATA / "annotations"
OUTPUT = ROOT / "outputs" / "annotated_samples"

SAMPLE_SIZE = 8
RANDOM_SEED = 42


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ANNOTATIONS.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {ANNOTATIONS}")

    rng = random.Random(RANDOM_SEED)
    samples = rng.sample(xml_files, min(SAMPLE_SIZE, len(xml_files)))

    for xml_path in samples:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")

        if not filename:
            print(f"Skipping {xml_path.name}: missing filename")
            continue

        image_path = IMAGES / filename
        if not image_path.exists():
            print(f"Skipping {xml_path.name}: image not found: {filename}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]

        for obj in root.findall("object"):
            label = obj.findtext("name", default="UNKNOWN")
            box = obj.find("bndbox")
            if box is None:
                continue

            try:
                x1 = int(float(box.findtext("xmin")))
                y1 = int(float(box.findtext("ymin")))
                x2 = int(float(box.findtext("xmax")))
                y2 = int(float(box.findtext("ymax")))
            except (TypeError, ValueError):
                print(f"Skipping invalid box in {xml_path.name}")
                continue

            # Clip coordinates to the image boundaries.
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        output_path = OUTPUT / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"Saved: {output_path}")

    print(f"\nDone. Open this folder:\n{OUTPUT}")


if __name__ == "__main__":
    main()
