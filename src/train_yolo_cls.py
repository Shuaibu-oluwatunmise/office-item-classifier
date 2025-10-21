"""
Train Multiple YOLO Models for Office Item Classification
Trains YOLOv8n, YOLOv8s, and YOLOv11 for systematic comparison
Automatically skips models that are already trained
"""
from ultralytics import YOLO
from pathlib import Path

# Get the absolute path to the project root
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data" / "processed"

# Define models to train
MODELS = {
    "yolov8n-cls": "yolov8n-cls.pt",  # Nano - smallest, fastest
    "yolov8s-cls": "yolov8s-cls.pt",  # Small - balanced
    "yolov11n-cls": "yolo11n-cls.pt", # Latest nano
    "yolov11s-cls": "yolo11s-cls.pt", # Latest small
}

trained_count = 0
skipped_count = 0

for model_name, pretrained_weights in MODELS.items():
    # Check if model already trained
    model_path = ROOT_DIR / "runs" / "classify" / f"{model_name}_train" / "weights" / "best.pt"
    
    if model_path.exists():
        print(f"\n{'='*60}")
        print(f"✅ SKIPPING {model_name} - Already trained!")
        print(f"Model found at: {model_path}")
        print(f"{'='*60}\n")
        skipped_count += 1
        continue
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(pretrained_weights)
    
    # Train
    model.train(
        data=str(DATA_DIR),
        epochs=25,
        imgsz=224,
        batch=32,
        device="cpu",
        workers=16, #adjust to your CPU capacities
        project=str(ROOT_DIR / "runs" / "classify"),
        name=f"{model_name}_train"
    )
    
    trained_count += 1
    print(f"\n{model_name} training complete!")
    print(f"Results saved to: runs/classify/{model_name}_train/\n")

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Models trained: {trained_count}")
print(f"Models skipped: {skipped_count}")
if trained_count > 0:
    print("ALL NEW MODELS TRAINED SUCCESSFULLY!")
else:
    print("All models were already trained!")
print("="*60)