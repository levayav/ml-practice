from pathlib import Path
import shutil

import fiftyone as fo
from fiftyone import ViewField as F
import yaml

import random


CLASSES_PATH = Path("configs/classes.txt")
OUTPUT_DIR = Path("data/processed")
DATA_YAML_PATH = Path("configs/data.yaml")

DATASET_DIR = Path('data/raw/OpenLogo')


def read_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError("configs/classes.txt not found. Run EDA step first.")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if not classes:
        raise ValueError("configs/classes.txt is empty")

    return classes


def prepare_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_split(sample):
    split = getattr(sample, "split", None)
    if split in ["train", "val", "test"]:
        return split
    return "train"


def convert_bbox_to_yolo(bbox):
    # FiftyOne bbox: [x_top_left, y_top_left, width, height], normalized
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2
    return x_center, y_center, w, h


def main():
    selected_classes = read_classes()
    class_to_id = {name: idx for idx, name in enumerate(selected_classes)}

    prepare_dirs()

    print("Loading OpenLogo")
    dataset = fo.Dataset.from_dir(dataset_dir=str(DATASET_DIR), dataset_type=fo.types.FiftyOneDataset)

    print("Filtering supervised subset")
    view = dataset.match(F("ground_truth") != None)

    train_samples = []
    test_samples = []

    for sample in view:
        if sample.split == "train":
            train_samples.append(sample)
        elif sample.split == "test":
            test_samples.append(sample)

    random.seed(42)
    random.shuffle(train_samples)

    val_size = int(len(train_samples) * 0.2)

    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]

    copied_images = 0
    written_labels = 0

    for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        for sample in samples:
            split = split_name
            detections = sample.ground_truth.detections if sample.ground_truth else []

            yolo_lines = []
            for det in detections:
                if det.label not in class_to_id:
                    continue

                class_id = class_to_id[det.label]
                x_center, y_center, w, h = convert_bbox_to_yolo(det.bounding_box)

                # Simple validation: skip obviously broken boxes
                if w <= 0 or h <= 0:
                    continue
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    continue

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                )

            if not yolo_lines:
                continue

            src_image = Path(sample.filepath)
            dst_image = OUTPUT_DIR / "images" / split / src_image.name
            dst_label = OUTPUT_DIR / "labels" / split / f"{src_image.stem}.txt"

            shutil.copy2(src_image, dst_image)
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            copied_images += 1
            written_labels += 1

    data_yaml = {
        "train": str(OUTPUT_DIR / "images" / "train"),
        "val": str(OUTPUT_DIR / "images" / "val"),
        "test": str(OUTPUT_DIR / "images" / "test"),
        "nc": len(selected_classes),
        "names": selected_classes,
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print("Done")
    print("Copied images:", copied_images)
    print("Written label files:", written_labels)
    print("Data config:", DATA_YAML_PATH)


if __name__ == "__main__":
    main()
