"""
Evaluate Multiple YOLO Models on Office Item Classification
Compares YOLOv8n, YOLOv8s, YOLOv11n, and YOLOv11s performance
"""

from ultralytics import YOLO
from pathlib import Path
import json

ROOT_DIR = Path(__file__).parent.parent.absolute()
TEST_DIR = ROOT_DIR / "data" / "processed" / "test"
RESULTS_DIR = ROOT_DIR / "results" / "yolo"

# Models to evaluate
MODELS = {
    "YOLOv8n-cls": ROOT_DIR / "runs" / "classify" / "yolov8n-cls_train" / "weights" / "best.pt",
    "YOLOv8s-cls": ROOT_DIR / "runs" / "classify" / "yolov8s-cls_train" / "weights" / "best.pt",
    "YOLOv11n-cls": ROOT_DIR / "runs" / "classify" / "yolov11n-cls_train" / "weights" / "best.pt",
    "YOLOv11s-cls": ROOT_DIR / "runs" / "classify" / "yolov11s-cls_train" / "weights" / "best.pt",
}

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("YOLO MODELS EVALUATION - Office Item Classification")
print("="*60)

all_results = {}

for model_name, model_path in MODELS.items():
    print(f"\n{'-'*60}")
    print(f"Evaluating {model_name}")
    print(f"{'-'*60}")
    
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        print(f"   Skipping {model_name}...")
        continue
    
    # Load model
    model = YOLO(str(model_path))
    
    # Evaluate
    metrics = model.val(data=str(TEST_DIR))
    
    # Store results
    all_results[model_name] = {
        "top1_accuracy": float(metrics.top1),
        "top5_accuracy": float(metrics.top5),
    }
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"  Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")

# Save comparison results
with open(RESULTS_DIR / "yolo_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print comparison table
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<20} {'Top-1 Acc':<15} {'Top-5 Acc':<15}")
print("-"*60)
for model_name, results in all_results.items():
    print(f"{model_name:<20} {results['top1_accuracy']*100:>6.2f}%        {results['top5_accuracy']*100:>6.2f}%")
print("="*60)

print(f"\nDetailed results saved to: {RESULTS_DIR}")
print("Individual confusion matrices available in runs/classify/[model_name]_train/")