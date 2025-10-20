"""
Train Multiple YOLO Models for Office Item Classification
Trains YOLOv8n, YOLOv8s, and YOLOv11 for systematic comparison
"""

from ultralytics import YOLO
import os

ROOT_DIR = r"C:\Users\shuai\office-item-classifier"
DATA_YAML = os.path.join(ROOT_DIR, "data", "yolo_data.yaml")

# Define models to train
MODELS = {
    "yolov8n-cls": "yolov8n-cls.pt",  # Nano - smallest, fastest
    "yolov8s-cls": "yolov8s-cls.pt",  # Small - balanced
    "yolov11n-cls": "yolo11n-cls.pt", # Latest nano
    "yolov11n-cls": "yolo11s-cls.pt", # Latest small
}

for model_name, pretrained_weights in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(pretrained_weights)
    
    # Train
    model.train(
        data=DATA_YAML,
        epochs=25,
        imgsz=224,
        batch=32,
        device="cpu",
        workers=0,
        project=os.path.join(ROOT_DIR, "runs", "classify"),
        name=f"{model_name}_train"
    )
    
    print(f"\n{model_name} training complete!")
    print(f"Results saved to: runs/classify/{model_name}_train/\n")

print("\n" + "="*60)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)