from pathlib import Path
from PIL import Image, ImageDraw

import fiftyone as fo


DATASET_DIR = Path('data/raw/OpenLogo')
OUTPUT_DIR = Path('results/predictions/day3_dataset_check')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def draw_detections(sample, output_path: Path):

    image = Image.open(sample.filepath).convert('RGB')
    draw = ImageDraw.Draw(image)

    width, height = image.size

    detections = sample.ground_truth.detections if sample.ground_truth else []

    for det in detections:
        x, y, w, h = det.bounding_box
        
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)

        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1 - 12)), det.label, fill='red')

    image.save(output_path)

def main():
    print('Loading local OpenLogo dataset')

    dataset = fo.Dataset.from_dir(dataset_dir=str(DATASET_DIR), dataset_type=fo.types.FiftyOneDataset)
    print('Dataset loaded')
    
    print(dataset)
    print('\nField schema:')
    print(dataset.schema)

    sample = dataset.first()
    print('\nFirst sample')
    print('filepath:', sample.filepath)
    print('split:', getattr(sample, 'split', None))
    print('supervision_level:', getattr(sample, 'supervision_level', None))
    print('num_logos:', getattr(sample, 'num_logos', None))
    print('logo_classes:', getattr(sample, 'logo_classes', None))
    print('ground_truth:', sample.ground_truth)

    print('\nSaving examples with bounding boxes')
    for idx, sample in enumerate(dataset.take(10)):
        output_path = OUTPUT_DIR / f'example_{idx:02d}.jpg'
        draw_detections(sample, output_path)
        print('saved', output_path)

    print('\nDone. Open results/predictions/day3_dataset_check/')

if __name__ == '__main__':
    main()