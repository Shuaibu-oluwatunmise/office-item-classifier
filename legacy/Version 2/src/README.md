# Source Scripts Documentation

**Version:** 2.0  
**Status:** ResNet18 complete (99.85% test accuracy), YOLO training in progress

This directory contains all Python scripts for the office item classification pipeline. Version 2.0 implements ResNet18 and 4 YOLO variants (v8n, v8s, v11n, v11s) for systematic architecture comparison.

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
- `data/processed/train/` - Training images (15,750 = 70%)
- `data/processed/val/` - Validation images (3,375 = 15%)
- `data/processed/test/` - Test images (3,375 = 15%)

**Dataset Stats:**
- Total: 22,500 images (2,500 per class)
- 9 classes perfectly balanced
- Stratified split maintains proportions

---

## Training Scripts

### `train.py` (ResNet18)
**Purpose:** Trains ResNet18 model with transfer learning.

**Status:** ✅ **COMPLETE - 99.85% Test Accuracy**

**Functionality:**
- Loads pretrained ResNet18 from ImageNet
- Replaces final fully connected layer for 9 classes
- Fine-tunes all layers (not frozen)
- Comprehensive data augmentation:
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
- Device: CPU (auto-detects CUDA if available)
- Workers: 16 (optimized for 20-core CPU)

**Training Performance:**
- Time: 115 minutes (20-core Intel Core Ultra 7 265K)
- Validation Accuracy: 99.88%
- Test Accuracy: **99.85%**
- 6 classes with perfect 100% F1-score

**Outputs:**
- `models/best_model.pth` - Model with highest validation accuracy
- `models/final_model.pth` - Model from last epoch
- `models/training_history.json` - All metrics per epoch

---

### `train_yolo_cls.py` (YOLO Models)
**Purpose:** Trains multiple YOLO classification models with smart skip logic.

**Status:** 🔄 **IN PROGRESS**
- ✅ YOLOv8n-cls: Complete (99.9% validation)
- 🔄 YOLOv8s-cls: Queued
- 🔄 YOLOv11n-cls: Queued
- 🔄 YOLOv11s-cls: Queued

**Functionality:**
- Trains 4 YOLO variants: YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s
- **Smart Skip Logic:** Automatically skips already-trained models
- Loads pretrained weights from Ultralytics
- Trains on organized dataset (directory-based, no YAML needed)
- Automatic data augmentation from YOLO framework
- Saves best model based on validation accuracy
- Generates training curves and metrics automatically

**Usage:**
```bash
python src/train_yolo_cls.py
```

**Configuration:**
- Models: 4 variants (nano and small for v8 and v11)
- Batch size: 32
- Epochs: 25
- Image size: 224×224
- Device: CPU
- Workers: 16

**Smart Features:**
- Checks for existing `best.pt` before training
- Skips completed models automatically
- Prints skip confirmation with model path
- Shows training summary (trained vs skipped counts)

**Outputs (per model):**
- `runs/classify/{model_name}_train/weights/best.pt` - Best model
- `runs/classify/{model_name}_train/weights/last.pt` - Last epoch
- Training curves, confusion matrix, results in `runs/` folder

**YOLOv8n Results:**
- Validation Accuracy: 99.9%
- Training Time: ~150 minutes
- Model Size: ~12 MB

---

## Evaluation Scripts

### `evaluate.py` (ResNet Evaluation)
**Purpose:** Evaluates trained ResNet18 model on test set.

**Status:** ✅ **COMPLETE**

**Functionality:**
- Loads best ResNet model from `models/best_model.pth`
- Runs inference on entire test set (3,375 images)
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

**Results Achieved:**
- Test Accuracy: **99.85%**
- Macro F1-Score: **0.9985**
- Perfect Classes (100% F1): 6 out of 9
- Worst Class: Laptop (99.60% - still excellent!)

**Outputs:**
- `results/test_metrics.json` - All metrics in JSON format
- `results/confusion_matrix.png` - Visual confusion matrix (12×10 figure)
- `results/classification_report.txt` - Detailed text report
- `results/per_class_metrics.csv` - Spreadsheet-format metrics
- `results/confusion_matrix.csv` - Raw confusion data

---

### `evaluate_yolo_cls.py` (YOLO Evaluation)
**Purpose:** Evaluates multiple trained YOLO models on test set.

**Status:** 📅 **READY** (run after all YOLO models trained)

**Functionality:**
- Loads all available YOLO models from `runs/classify/` folders
- Skips models that don't exist yet (graceful handling)
- Runs validation on test directory
- Calculates Top-1 and Top-5 accuracy for each model
- Creates comparison summary across all models
- Saves metrics to `results/yolo/` folder

**Usage:**
```bash
python src/evaluate_yolo_cls.py
```

**Features:**
- Evaluates: YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s
- Comparison table with all models
- JSON export for further analysis
- Individual confusion matrices in training folders

**Outputs:**
- `results/yolo/yolo_comparison.json` - All model metrics
- Comparison table printed to console
- Individual results available in `runs/classify/{model}_train/`

---

### `model_competition.py` 🏆 **NEW!**
**Purpose:** Ultimate showdown - compares all 5 models on excesses dataset.

**Status:** 📅 **READY** (run after all models trained)

**Functionality:**
- Tests ResNet18 + 4 YOLO models on same dataset
- Uses `excesses/` folder (~11K additional images)
- Comprehensive metrics for each model:
  - Overall accuracy
  - Per-class accuracy heatmap
  - Average confidence scores
  - Correct vs incorrect prediction confidence
- Creates championship visualizations:
  - Accuracy bar chart with rankings
  - Confidence comparison
  - Per-class performance heatmap
  - Correct/incorrect confidence analysis
- Generates detailed leaderboard with rankings (🥇🥈🥉)
- Crowns the champion model!

**Usage:**
```bash
python src/model_competition.py
```

**Features:**
- Tests all 5 models on identical unseen data
- Head-to-head comparison
- Visual competition dashboard (4-panel chart)
- Leaderboard CSV export
- Detailed championship report

**Outputs:**
- `results/competition/model_competition_results.png` - 4-panel visualization
- `results/competition/leaderboard.csv` - Rankings table
- `results/competition/competition_report.txt` - Championship summary
- `results/competition/detailed_results.json` - Raw metrics

---

## Inference Scripts

### `inference.py`
**Purpose:** Classifies single image files using trained ResNet model.

**Functionality:**
- Accepts image file path as command-line argument
- Loads best ResNet model (99.85% accuracy)
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
🎯 Predicted Class: Monitor
📊 Confidence: 99.87%

Top 3 Predictions:
1. Monitor              ████████████████████  99.87%
2. Laptop               █░░░░░░░░░░░░░░░░░░░   0.08%
3. Mobile Phone         ░░░░░░░░░░░░░░░░░░░░   0.03%
```

**Supported Formats:** JPEG, PNG, BMP

**Performance:** ~100-150ms per image on CPU

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
- **'s'** - Save screenshot with timestamp

**Performance:**
- ~8-12 FPS on laptop CPU (Intel i7-1255U)
- ~15-20 FPS on desktop CPU (Intel Core Ultra 7 265K)
- Best results with isolated objects
- Good lighting improves accuracy

**Tips:**
- Hold objects close to camera for best results
- Use plain backgrounds when possible
- Ensure good lighting conditions
- Model performs best on objects similar to training data

---

## Archive Folder

### `archive/`
Contains legacy scripts from Version 1.0 exploration phase:

**Scripts:**
- `download_datasets.py` - Manual Roboflow download helper
- `download_kaggle_datasets.py` - Automated Kaggle downloader (experimental)
- `inspect_downloads.py` - Dataset structure explorer
- `smart_organize.py` - Early organization script (deprecated)

**Status:** Archived for reference, not used in v2.0

---

## Dependencies

**Core Libraries:**
- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- ultralytics 8.0+ (for YOLO)
- OpenCV (cv2) - for camera inference
- scikit-learn - for evaluation metrics
- matplotlib, seaborn - for visualizations
- pandas, numpy - for data handling
- tqdm - for progress bars

**Installation:**
```bash
pip install -r requirements.txt
```

See `requirements.txt` in project root for complete dependency list.

---

## Development Workflow

**1. Data Preparation:**
```bash
# Organize images in data/raw/[class_name]/ folders
python src/organize_dataset.py
```

**2. Training Phase:**
```bash
# Train ResNet (~2 hours on 20-core CPU)
python src/train.py

# Train YOLO models (~10 hours total for all 4)
python src/train_yolo_cls.py  # Smart skip logic included!
```

**3. Evaluation Phase:**
```bash
# Evaluate ResNet
python src/evaluate.py

# Evaluate all YOLO models
python src/evaluate_yolo_cls.py

# Ultimate model competition
python src/model_competition.py
```

**4. Inference/Deployment:**
```bash
# Single image
python src/inference.py path/to/image.jpg

# Live webcam
python src/camera_inference.py
```

---

## Technical Performance

**Training Times (20-core CPU):**
- ResNet18: 115 minutes (25 epochs)
- YOLOv8n: ~150 minutes (25 epochs)
- YOLOv8s: ~TBD
- YOLOv11n: ~TBD
- YOLOv11s: ~TBD

**Model Sizes:**
- ResNet18: ~44 MB
- YOLOv8n-cls: ~12 MB (3.7× smaller)
- YOLOv8s-cls: ~TBD
- YOLOv11n: ~TBD
- YOLOv11s: ~TBD

**Inference Speed (CPU):**
- ResNet18: ~100-150ms per image
- YOLO models: ~30-80ms per image (faster)

**Accuracy Achieved:**
- ResNet18 Test: **99.85%** ✅
- YOLOv8n Val: **99.9%** ✅
- Competition winner: TBD 🏆

---

## Script Execution Order

**For complete pipeline from scratch:**
```bash
# 1. Organize data
python src/organize_dataset.py

# 2. Train models (can run in parallel on different machines)
python src/train.py              # ResNet
python src/train_yolo_cls.py     # All YOLO models

# 3. Evaluate models
python src/evaluate.py           # ResNet evaluation
python src/evaluate_yolo_cls.py  # YOLO evaluation

# 4. Model competition (after all trained)
python src/model_competition.py  # Champion selection

# 5. Use best model for inference
python src/inference.py test_image.jpg
python src/camera_inference.py
```

---

## Version History

**Version 1.0 (Archived):**
- Single architecture: ResNet18 only
- Test accuracy: 96.37%
- Documentation: `legacy/src_README_v1.md`

**Version 2.0 (Current):**
- 5 models: ResNet18 + 4 YOLO variants
- ResNet test accuracy: **99.85%** ✅
- Smart training with skip logic
- Ultimate model competition framework
- Deployment-ready performance

---

*Last Updated: October 21, 2024*  
*Status: ResNet complete (99.85%), YOLO training in progress, competition framework ready*