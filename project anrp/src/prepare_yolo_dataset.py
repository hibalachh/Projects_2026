
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "archive"
IMAGES = SOURCE / "images"
ANNOTATIONS = SOURCE / "annotations"
OUTPUT = ROOT / "data" / "processed" / "license_plates"

SEED = 42
CLASS_ID = 0
CLASS_NAME = "licence"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_annotation(xml_path, image_width, image_height):
    """Convert Pascal VOC boxes into normalized YOLO labels."""
    root = ET.parse(xml_path).getroot()
    labels = []

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name.lower() != CLASS_NAME:
            raise ValueError(
                f"Unexpected label {name!r} in {xml_path.name}"
            )

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {xml_path.name}")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        if not (
            0 <= xmin < xmax <= image_width
            and 0 <= ymin < ymax <= image_height
        ):
            raise ValueError(
                f"Invalid box in {xml_path.name}: "
                f"{xmin, ymin, xmax, ymax}"
            )

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        labels.append(
            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    if not labels:
        raise ValueError(f"No labeled objects in {xml_path.name}")

    return labels


def main():
    if not IMAGES.is_dir() or not ANNOTATIONS.is_dir():
        raise FileNotFoundError(f"Dataset folders not found under {SOURCE}")

    image_by_stem = {
        p.stem: p
        for p in IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    xml_by_stem = {p.stem: p for p in ANNOTATIONS.glob("*.xml")}

    if set(image_by_stem) != set(xml_by_stem):
        missing_xml = sorted(set(image_by_stem) - set(xml_by_stem))
        missing_image = sorted(set(xml_by_stem) - set(image_by_stem))
        raise ValueError(
            f"Image/XML mismatch. Missing XML: {missing_xml[:10]}; "
            f"missing images: {missing_image[:10]}"
        )

    # Validate and convert every annotation before creating the output.
    prepared = {}
    for stem, image_path in image_by_stem.items():
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")

        height, width = image.shape[:2]
        xml_path = xml_by_stem[stem]
        prepared[stem] = (
            image_path,
            parse_annotation(xml_path, width, height),
        )

    # Reproducible shuffle and split.
    stems = sorted(prepared)
    random.Random(SEED).shuffle(stems)

    n_train = int(0.8 * len(stems))
    n_val = int(0.1 * len(stems))

    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    # Only clear the generated split folders, not the raw dataset.
    if OUTPUT.exists():
        answer = input(
            f"{OUTPUT} already exists. Rebuild its generated contents? (y/n): "
        ).strip().lower()
        if answer != "y":
            print("Cancelled; existing output was not changed.")
            return

    for split in splits:
        for kind in ("images", "labels"):
            folder = OUTPUT / kind / split
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    for split, split_stems in splits.items():
        for stem in split_stems:
            image_path, labels = prepared[stem]

            shutil.copy2(
                image_path,
                OUTPUT / "images" / split / image_path.name,
            )

            label_path = OUTPUT / "labels" / split / f"{stem}.txt"
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    # Use an absolute path so the YAML works from any current directory.
    dataset_yaml = {
        "path": OUTPUT.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {CLASS_ID: CLASS_NAME},
    }

    with (OUTPUT / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    print("\n=== YOLO DATASET PREPARATION ===")
    for split, split_stems in splits.items():
        object_count = sum(len(prepared[s][1]) for s in split_stems)
        print(
            f"{split:5}: {len(split_stems):3} images, "
            f"{object_count:3} annotated plates"
        )

    print(f"\nTotal images: {len(stems)}")
    print(f"Dataset YAML: {OUTPUT / 'dataset.yaml'}")
    print("Dataset preparation completed successfully.")


if __name__ == "__main__":
    main()
