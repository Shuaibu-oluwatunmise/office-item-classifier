"""
Evaluate YOLOv8 Classification Model on Office Item Dataset
Generates comprehensive metrics comparable to ResNet evaluation
"""

from ultralytics import YOLO
import os
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\shuai\office-item-classifier")
MODEL_PATH = ROOT_DIR / "runs" / "classify" / "yolo_cls_train" / "weights" / "best.pt"
TEST_DIR = ROOT_DIR / "data" / "processed" / "test"
RESULTS_DIR = ROOT_DIR / "results" / "yolo"

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("YOLO-CLS EVALUATION - Office Item Classification")
print("=" * 60)

# Load trained model
print(f"\nLoading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Evaluate on test set
print("\nRunning evaluation on test set...")
metrics = model.val(data=str(TEST_DIR))

# Print results
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
print(f"Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")
print("=" * 60)

# Save metrics
with open(RESULTS_DIR / "yolo_test_metrics.txt", "w") as f:
    f.write("YOLO-CLS Test Set Evaluation\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: YOLOv8n-cls\n")
    f.write(f"Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)\n")
    f.write(f"Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)\n")
    f.write("=" * 60 + "\n")

print(f"\nResults saved to: {RESULTS_DIR}")
print("\nNote: YOLO outputs confusion matrix automatically during training.")
print("Check: runs/classify/yolo_cls_train/ for confusion matrix and other metrics")