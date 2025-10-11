# Source Scripts Documentation

**Version:** 2.0  
**Status:** Retraining with improved dataset and dual architecture approach

This directory contains all Python scripts for the office item classification pipeline. Version 2.0 implements both ResNet18 and YOLOv8-cls for systematic architecture comparison.

---

## Core Pipeline Scripts

### `organize_dataset.py`
**Purpose:** Splits raw images into train/validation/test sets with stratified sampling.

**Functionality:**
- Reads all images from `data/raw/[class_name]/` folders
- Creates stratified 70/15/15 split maintaining class proportions
- Outputs organized folders to `data/processed/`
- Uses random seed 42 for reproducibility

**Usage:**
```bash
python src/organize_dataset.py
```

**Outputs:**
- `data/processed/train/` - Training images (70%)
- `data/processed/val/` - Validation images (15%)
- `data/processed/test/` - Test images (15%)

**Technical Details:**
- Preserves class distribution across all splits
- Handles multiple image formats (JPEG, PNG)
- Prints statistics for verification

---

## Training Scripts

### `train.py` (ResNet18)
**Purpose:** Trains ResNet18 model with transfer learning.

**Functionality:**
- Loads pretrained ResNet18 from ImageNet
- Replaces final fully connected layer for 11 classes
- Fine-tunes all layers (not frozen)
- Applies comprehensive data augmentation:
  - Random crop (224×224)
  - Random horizontal flip
  - Color jitter (brightness, contrast, saturation ±20%)
  - Random rotation (±15°)
  - ImageNet normalization
- Saves best model based on validation accuracy
- Tracks training history (loss, accuracy per epoch)

**Usage:**
```bash
python src/train.py
```

**Configuration:**
- Optimizer: Adam (LR=0.001)
- Batch size: 32
- Epochs: 25
- Device: Automatic (CUDA if available, else CPU)

**Outputs:**
- `models/best_model.pth` - Model with highest validation accuracy
- `models/final_model.pth` - Model from last epoch
- `models/training_history.json` - All metrics per epoch

---

### `train_yolo_cls.py` (YOLOv8)
**Purpose:** Trains YOLOv8 nano classification model.

**Functionality:**
- Loads pretrained YOLOv8n-cls from Ultralytics
- Trains on organized dataset using YAML configuration
- Automatic data augmentation from YOLO framework
- Saves best model based on validation accuracy
- Generates training curves and metrics automatically

**Usage:**
```bash
python src/train_yolo_cls.py
```

**Configuration:**
- Model: YOLOv8n-cls (nano - smallest, fastest)
- Batch size: 32
- Epochs: 25
- Image size: 224×224
- Device: CPU (configurable)

**Outputs:**
- `runs/classify/yolo_cls_train/weights/best.pt` - Best model
- `runs/classify/yolo_cls_train/weights/last.pt` - Last epoch
- Training curves, confusion matrix, and metrics in `runs/` folder

---

## Evaluation Scripts

### `evaluate.py` (ResNet Evaluation)
**Purpose:** Evaluates trained ResNet18 model on test set.

**Functionality:**
- Loads best ResNet model from `models/best_model.pth`
- Runs inference on entire test set
- Calculates comprehensive metrics:
  - Overall accuracy
  - Macro F1-score (average across all classes)
  - Per-class precision, recall, F1-score
  - Confusion matrix
- Generates visualizations:
  - Confusion matrix heatmap
  - Classification report
- Saves all results to `results/` folder

**Usage:**
```bash
python src/evaluate.py
```

**Outputs:**
- `results/test_metrics.json` - All metrics in JSON format
- `results/confusion_matrix.png` - Visual confusion matrix
- `results/classification_report.txt` - Detailed text report
- `results/per_class_metrics.csv` - Spreadsheet-format metrics
- `results/confusion_matrix.csv` - Raw confusion data

---

### `evaluate_yolo_cls.py` (YOLO Evaluation)
**Purpose:** Evaluates trained YOLOv8-cls model on test set.

**Functionality:**
- Loads best YOLO model from `runs/classify/yolo_cls_train/weights/best.pt`
- Runs validation on test directory
- Calculates Top-1 and Top-5 accuracy
- Saves metrics to `results/yolo/` folder
- Note: YOLO generates confusion matrix automatically during training

**Usage:**
```bash
python src/evaluate_yolo_cls.py
```

**Outputs:**
- `results/yolo/yolo_test_metrics.txt` - Test accuracy metrics
- Confusion matrix available in `runs/classify/yolo_cls_train/`

---

## Inference Scripts

### `inference.py`
**Purpose:** Classifies single image files using trained ResNet model.

**Functionality:**
- Accepts image file path as command-line argument
- Loads best ResNet model
- Preprocesses image (resize, normalize)
- Returns predicted class with confidence score
- Shows top 3 predictions with visual progress bars
- Provides confidence warnings for ambiguous predictions

**Usage:**
```bash
python src/inference.py path/to/image.jpg
```

**Example Output:**
```
🎯 Predicted Class: Mug
📊 Confidence: 98.45%

Top 3 Predictions:
1. Mug                  ████████████████████  98.45%
2. Water Bottle         ███░░░░░░░░░░░░░░░░░   1.23%
3. Mobile Phone         ░░░░░░░░░░░░░░░░░░░░   0.32%
```

**Supported Formats:** JPEG, PNG

---

### `camera_inference.py`
**Purpose:** Real-time classification from webcam using trained ResNet model.

**Functionality:**
- Captures live video from webcam using OpenCV
- Runs inference on each frame
- Displays prediction overlay with:
  - Current prediction
  - Confidence percentage
  - Confidence bar visualization
  - FPS counter
- Saves screenshots on demand
- Simplified display for better UX

**Usage:**
```bash
python src/camera_inference.py
```

**Controls:**
- **'q'** - Quit application
- **'s'** - Save screenshot

**Performance:**
- ~15-20 FPS on CPU (Intel i7-1255U)
- Best results with isolated objects on plain backgrounds
- Good lighting improves accuracy

**Tips:**
- Hold objects close to camera
- Use plain backgrounds for best results
- Ensure good lighting conditions

---

## Archive Folder

### `archive/`
Contains legacy scripts from initial dataset exploration phase (Version 1.0):

**Scripts:**
- `download_datasets.py` - Helper for manual Roboflow downloads
- `download_kaggle_datasets.py` - Automated Kaggle dataset downloader (attempted)
- `inspect_downloads.py` - Dataset structure explorer
- `smart_organize.py` - Early organization script (deprecated)

**Status:** These scripts are no longer used but kept for reference and project history documentation.

**Note:** Version 1.0 used manual Roboflow downloads. These Kaggle scripts were experimental and not used in final v1.0 dataset.

---

## Dependencies

**Core Libraries:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- ultralytics (for YOLO)
- OpenCV (cv2) - for camera inference
- scikit-learn - for evaluation metrics
- matplotlib, seaborn - for visualizations
- pandas, numpy - for data handling
- tqdm - for progress bars

**Installation:**
```bash
pip install -r requirements.txt
```

See `requirements.txt` in project root for complete dependency list with versions.

---

## Version History

**Version 1.0 (Archived):**
- Single architecture: ResNet18 only
- Dataset: Roboflow Universe (13,616 images)
- Results: 96.37% test accuracy
- Limitation: Poor real-world camera performance
- Documentation: See `legacy/src_README_v1.md`

**Version 2.0 (Current):**
- Dual architecture: ResNet18 + YOLOv8n-cls
- Dataset: Improved quality, diverse sources (in progress)
- Goal: 85-90% real-world accuracy
- Focus: Bridge domain gap for deployment

---

## Development Workflow

**Data Preparation:**
1. Collect diverse, high-quality images
2. Organize in `data/raw/[class_name]/` folders
3. Run `organize_dataset.py` to create splits

**Training Phase:**
1. Train ResNet: `python src/train.py` (~11 hours)
2. Train YOLO: `python src/train_yolo_cls.py` (~8-9 hours)

**Evaluation Phase:**
1. Evaluate ResNet: `python src/evaluate.py`
2. Evaluate YOLO: `python src/evaluate_yolo_cls.py`
3. Compare results, select best model

**Deployment:**
1. Use `inference.py` for file-based classification
2. Use `camera_inference.py` for real-time webcam

---

## Technical Notes

**Training Performance:**
- ResNet training: ~11 hours on CPU (25 epochs)
- YOLO training: ~8-9 hours on CPU (25 epochs)
- GPU significantly reduces training time

**Model Sizes:**
- ResNet18: 44.8 MB
- YOLOv8n-cls: ~12 MB (3.7× smaller)

**Inference Speed (CPU):**
- ResNet18: ~50-70ms per image
- YOLOv8n-cls: ~30-40ms per image (1.5-2× faster)

---

*Last Updated: October 10, 2025*  
*Version 2.0 - Dual architecture implementation with improved dataset*