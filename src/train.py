# RUN ONLY IF TRAINING NEW BASELINE MODEL
import os
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'archive', 'data.yaml'),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    mosaic=1.0,
    mixup=0.1,
    name='baseline_model'
)