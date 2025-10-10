# Source Scripts Documentation

This directory contains all scripts for the office item classification project.

## Core Pipeline Scripts

### `organize_dataset.py` ✅
**Status:** Complete  
**Purpose:** Splits raw images into train/validation/test sets with stratified sampling.  
**Usage:** 
```bash
python src/organize_dataset.py
```
**Details:**
- Split ratio: 70% train, 15% validation, 15% test
- Random seed: 42 (for reproducibility)
- Preserves class distribution across splits
- Output: `data/processed/train/`, `data/processed/val/`, `data/processed/test/`

**Results:**
- Training: 10,005 images
- Validation: 2,473 images  
- Test: 2,482 images
- Total: 14,960 images across 11 classes

---

### `train.py` ✅
**Status:** Complete  
**Purpose:** Trains ResNet18 model with transfer learning on office items dataset.

**Usage:**
```bash
python src/train.py
```

**Architecture:**
- Base model: ResNet18 (pretrained on ImageNet)
- Modified final layer for 11 classes
- Fine-tuned all layers

**Training Configuration:**
- Optimizer: Adam (LR=0.001)
- Batch size: 32
- Epochs: 25
- Device: CPU (Intel i7-1255U)
- Duration: 10 hours 48 minutes

**Data Augmentation (Training Only):**
- Random crop (224×224)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation ±20%)
- Random rotation (±15°)
- ImageNet normalization

**Outputs:**
- `models/best_model.pth` - Model with best validation accuracy (97.45%)
- `models/final_model.pth` - Model from final epoch
- `models/training_history.json` - All metrics per epoch

**Results:**
- Best validation accuracy: 97.45% (Epoch 22)
- Final validation accuracy: 95.99% (Epoch 25)
- Training accuracy: 97.50%
- Minimal overfitting: 0.05% gap

---

### `evaluate.py` ✅
**Status:** Complete  
**Purpose:** Evaluates trained model on test set and generates comprehensive metrics.

**Usage:**
```bash
python src/evaluate.py
```

**Metrics Calculated:**
- Overall accuracy
- Macro F1-score (average across all classes)
- Per-class precision, recall, F1-score
- Confusion matrix
- Classification report

**Outputs:**
- `results/test_metrics.json` - All metrics in JSON format
- `results/classification_report.txt` - Detailed text report
- `results/per_class_metrics.csv` - Per-class metrics table
- `results/confusion_matrix.csv` - Raw confusion matrix data
- `results/confusion_matrix.png` - Confusion matrix visualization

**Test Set Results:**
- Test accuracy: 96.37% (2,482 images)
- Macro F1-score: 0.9528
- Best class: office_chair (99.29% F1)
- Challenging class: computer_mouse (85.57% F1)

---

## Future Scripts (To Be Implemented)

### `inference.py` 🚧
**Status:** To be created  
**Purpose:** Run inference on single image files.

**Planned Usage:**
```bash
python src/inference.py path/to/image.jpg
```

**Expected Output:**
```
Predicted: mug
Confidence: 0.9523 (95.23%)
```

**Features:**
- Load best trained model
- Preprocess input image
- Return predicted class and confidence
- Handle various image formats (JPEG, PNG)

---

### `camera_inference.py` 🚧
**Status:** To be created  
**Purpose:** Real-time classification from webcam feed.

**Planned Usage:**
```bash
python src/camera_inference.py
```

**Features:**
- Live video capture
- Real-time prediction overlay
- Display confidence scores
- Press 'q' to quit

---

## Archived Scripts (Not Needed)

These scripts were used during initial data collection and can be removed:

- ~~`download_datasets.py`~~ - Initial setup helper
- ~~`download_kaggle_datasets.py`~~ - Kaggle API downloader
- ~~`inspect_downloads.py`~~ - Dataset structure explorer
- ~~`smart_organize.py`~~ - Legacy organizer

**Recommendation:** Delete or move to `archive/` folder.

---

## Script Dependencies

**Core Libraries:**
- PyTorch 2.0+
- torchvision
- scikit-learn (for metrics)
- matplotlib (for visualizations)
- seaborn (for confusion matrix)
- pandas (for CSV exports)
- tqdm (for progress bars)
- OpenCV (for camera inference)

**See `requirements.txt` in project root for complete list.**

---

## Development Workflow

**Completed Steps:**
1. ✅ Dataset collection (13,616 images from Roboflow Universe)
2. ✅ Dataset organization (`organize_dataset.py`)
3. ✅ Model training (`train.py`) - 97.45% validation accuracy
4. ✅ Model evaluation (`evaluate.py`) - 96.37% test accuracy

**Next Steps:**
5. 🚧 Single image inference (`inference.py`)
6. 🚧 Live camera inference (`camera_inference.py`)
7. 🚧 Error analysis documentation
8. 🚧 Code walkthrough video (2 minutes)

---

## Notes

**Training Performance:**
- Training on CPU took ~11 hours for 25 epochs
- GPU would significantly reduce training time
- Best model achieved at Epoch 22 (early stopping would save time)

**Model Size:**
- Model file: 44.8 MB
- Suitable for deployment on edge devices
- Fast inference even on CPU

**Test Set Balance:**
- All classes have 100-500 test images
- Largest: office_bin (467 images)
- Smallest: computer_mouse (110 images)
- Well-balanced for evaluation

---

*Last Updated: October 10, 2025*  
*All core scripts complete - ready for inference implementation*