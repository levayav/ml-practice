from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F

DATASET_DIR = Path('data/raw/OpenLogo')

PLOTS_DIR = Path("results/plots")
METRICS_DIR = Path("results/metrics")
CONFIGS_DIR = Path("configs")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def collect_stats(view):
    class_counter = Counter()
    bbox_rows = []
    objects_per_image = []

    for sample in view:
        detections = sample.ground_truth.detections if sample.ground_truth else []
        objects_per_image.append(len(detections))

        for det in detections:
            label = det.label
            x, y, w, h = det.bounding_box
            class_counter[label] += 1
            bbox_rows.append({
                "label": label,
                "bbox_width_norm": w,
                "bbox_height_norm": h,
                "bbox_area_norm": w * h,
                "split": getattr(sample, "split", None),
            })

    class_df = pd.DataFrame(
        [{"label": k, "objects": v} for k, v in class_counter.items()]
    ).sort_values("objects", ascending=False)

    bbox_df = pd.DataFrame(bbox_rows)
    objects_df = pd.DataFrame({"objects_per_image": objects_per_image})

    return class_df, bbox_df, objects_df


def save_plots(class_df, bbox_df, objects_df):
    top_classes = class_df.head(32)

    plt.figure(figsize=(12, 7))
    plt.bar(top_classes["label"], top_classes["objects"])
    plt.xticks(rotation=90)
    plt.title("Top logo classes by number of objects")
    plt.xlabel("Logo class")
    plt.ylabel("Object count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution_top32.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(bbox_df["bbox_area_norm"], bins=50)
    plt.title("Bounding box area distribution")
    plt.xlabel("Normalized bbox area")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bbox_area_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(objects_df["objects_per_image"], bins=30)
    plt.title("Objects per image distribution")
    plt.xlabel("Objects per image")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "objects_per_image_distribution.png", dpi=150)
    plt.close()


def main():
    print("Loading OpenLogo sample/dataset")
    dataset = fo.Dataset.from_dir(dataset_dir=str(DATASET_DIR), dataset_type=fo.types.FiftyOneDataset)

    print("Filtering supervised subset")
    supervised_view = dataset.match(F("ground_truth") != "None")

    print("Collecting stats")
    class_df, bbox_df, objects_df = collect_stats(supervised_view)

    class_df.to_csv(METRICS_DIR / "class_distribution.csv", index=False)
    bbox_df.to_csv(METRICS_DIR / "bbox_stats.csv", index=False)
    objects_df.to_csv(METRICS_DIR / "objects_per_image.csv", index=False)

    print("Saving plots")
    save_plots(class_df, bbox_df, objects_df)

    selected_classes = class_df.head(32)["label"].tolist()
    with open(CONFIGS_DIR / "classes.txt", "w", encoding="utf-8") as f:
        for cls in selected_classes:
            f.write(cls + "\n")

    print("Selected classes:")
    for cls in selected_classes:
        print("-", cls)

    print("\nDone. Check results/metrics/, results/plots/, configs/classes.txt")


if __name__ == "__main__":
    main()
