
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

CSV_PATH = Path("outputs/anpr/paddleocr_anpr_results.csv")
OUTPUT_PATH = Path("outputs/evaluation/problem_crops_contact_sheet.jpg")

target_images = {
    "Cars100", "Cars109", "Cars138", "Cars199", "Cars200",
    "Cars210", "Cars19", "Cars330", "Cars35", "Cars357",
    "Cars398", "Cars421",
}

df = pd.read_csv(CSV_PATH)
df["image_key"] = df["image"].astype(str).apply(
    lambda value: Path(value).stem
)

rows = df[
    df["image_key"].isin(target_images)
    & df["crop_path"].notna()
].copy()

if rows.empty:
    raise SystemExit("No matching crops found. Check the CSV paths.")

cell_width = 420
cell_height = 145
columns = 2
rows_count = (len(rows) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows_count * cell_height),
    "white",
)
draw = ImageDraw.Draw(sheet)

for index, (_, row) in enumerate(rows.iterrows()):
    crop_path = Path(str(row["crop_path"]))

    if not crop_path.exists():
        crop_path = Path("outputs") / crop_path

    x = (index % columns) * cell_width
    y = (index // columns) * cell_height

    draw.text(
        (x + 8, y + 5),
        f'{row["image"]} | {row["plate_number"]} '
        f'| YOLO={row["detection_confidence"]:.2f}',
        fill="black",
    )

    if crop_path.exists():
        image = Image.open(crop_path).convert("RGB")
        image.thumbnail((cell_width - 16, cell_height - 38))
        sheet.paste(image, (x + 8, y + 30))
    else:
        draw.text(
            (x + 8, y + 40),
            f"Missing crop: {crop_path}",
            fill="red",
        )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT_PATH, quality=95)

print(f"Inspected crops: {len(rows)}")
print(f"Saved contact sheet: {OUTPUT_PATH}")
