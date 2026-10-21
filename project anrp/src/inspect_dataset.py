
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "archive"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not IMAGES_DIR.exists() or not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(
            f"Expected folders:\n{IMAGES_DIR}\n{ANNOTATIONS_DIR}"
        )

    images = {
        p.stem: p
        for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    xml_files = {
        p.stem: p
        for p in ANNOTATIONS_DIR.glob("*.xml")
    }

    labels = Counter()
    missing_xml = sorted(set(images) - set(xml_files))
    missing_images = sorted(set(xml_files) - set(images))
    invalid_xml = []
    invalid_boxes = []
    total_objects = 0

    for stem, xml_path in xml_files.items():
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            invalid_xml.append((xml_path.name, str(exc)))
            continue

        size = root.find("size")
        width = int(size.findtext("width", "0")) if size is not None else 0
        height = int(size.findtext("height", "0")) if size is not None else 0

        objects = root.findall("object")
        total_objects += len(objects)

        for index, obj in enumerate(objects, start=1):
            label = (obj.findtext("name") or "UNKNOWN").strip()
            labels[label] += 1

            box = obj.find("bndbox")
            if box is None:
                invalid_boxes.append((xml_path.name, index, "Missing bndbox"))
                continue

            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                if (
                    width <= 0 or height <= 0
                    or xmin < 0 or ymin < 0
                    or xmax <= xmin or ymax <= ymin
                    or xmax > width or ymax > height
                ):
                    invalid_boxes.append(
                        (xml_path.name, index,
                         f"Invalid box {(xmin, ymin, xmax, ymax)} "
                         f"for image size {(width, height)}")
                    )
            except (TypeError, ValueError):
                invalid_boxes.append(
                    (xml_path.name, index, "Missing or nonnumeric coordinates")
                )

    print("\n=== ANPR DATASET REPORT ===")
    print(f"Dataset folder:       {DATASET_DIR}")
    print(f"Images found:         {len(images)}")
    print(f"XML files found:      {len(xml_files)}")
    print(f"Annotated objects:    {total_objects}")
    print(f"Labels:               {dict(labels)}")
    print(f"Images without XML:   {len(missing_xml)}")
    print(f"XML without image:    {len(missing_images)}")
    print(f"Invalid XML files:    {len(invalid_xml)}")
    print(f"Invalid boxes:        {len(invalid_boxes)}")

    if missing_xml:
        print("\nFirst images without XML:", missing_xml[:10])
    if missing_images:
        print("\nFirst XML files without image:", missing_images[:10])
    if invalid_xml:
        print("\nFirst XML parsing errors:", invalid_xml[:5])
    if invalid_boxes:
        print("\nFirst invalid boxes:", invalid_boxes[:5])


if __name__ == "__main__":
    main()
