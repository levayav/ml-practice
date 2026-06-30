from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    model.train(
        data='configs/data.yaml',
        epochs=10,
        imgsz=640,
        batch=8,
        project='results/training',
        name='yolov8n_baseline'
    )

if __name__ == '__main__':
    main()