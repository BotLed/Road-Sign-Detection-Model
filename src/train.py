# RUN ONLY IF TRAINING NEW BASELINE MODEL
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='/content/datasets/archive/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    mosaic=1.0,
    mixup=0.1,
    name='baseline_model'
)