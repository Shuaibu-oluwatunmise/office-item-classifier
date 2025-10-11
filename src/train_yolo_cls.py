"""
Train YOLOv8 for Office Item Classification
"""

from ultralytics import YOLO
import os

ROOT_DIR = r"C:\Users\shuai\office-item-classifier"
DATA_YAML = os.path.join(ROOT_DIR, "data", "yolo_data.yaml")

# Load YOLOv8 classification model (n = nano, fastest)
model = YOLO("yolov8n-cls.pt")

# Train on CPU
model.train(
    data=DATA_YAML,
    epochs=25,
    imgsz=224,
    batch=32,
    device="cpu",
    workers=0,  # prevent multiprocessing issues on Windows
    project=os.path.join(ROOT_DIR, "runs"),
    name="yolo_cls_train"
)
