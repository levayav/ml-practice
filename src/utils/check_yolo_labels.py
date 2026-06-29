from pathlib import Path
import random

from PIL import Image, ImageDraw
import yaml


DATA_DIR = Path("data/processed")
DATA_YAML = Path("configs/data.yaml")
OUTPUT_DIR = Path("results/predictions/day5_yolo_label_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_class_names():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


def draw_yolo_boxes(image_path, label_path, class_names, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        class_id, x_center, y_center, box_w, box_h = line.split()
        class_id = int(class_id)
        x_center = float(x_center)
        y_center = float(y_center)
        box_w = float(box_w)
        box_h = float(box_h)

        x1 = int((x_center - box_w / 2) * width)
        y1 = int((y_center - box_h / 2) * height)
        x2 = int((x_center + box_w / 2) * width)
        y2 = int((y_center + box_h / 2) * height)

        label = class_names[class_id]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), label, fill="red")

    image.save(output_path)


def main():
    class_names = load_class_names()

    image_dir = DATA_DIR / "images" / "train"
    label_dir = DATA_DIR / "labels" / "train"

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError("No train images found in data/processed/images/train")

    selected_images = random.sample(images, min(10, len(images)))

    for idx, image_path in enumerate(selected_images):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print("Missing label:", label_path)
            continue

        output_path = OUTPUT_DIR / f"check_{idx:02d}_{image_path.name}"
        draw_yolo_boxes(image_path, label_path, class_names, output_path)
        print("saved", output_path)

    print("Done. Open results/predictions/day5_yolo_label_check/")


if __name__ == "__main__":
    main()
