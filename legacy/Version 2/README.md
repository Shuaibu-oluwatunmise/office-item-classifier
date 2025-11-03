# Office Item Classifier

**Module:** PDE3802 - AI in Robotics  
**Assessment:** Part B - Office-Goods Classification Code  
**Author:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Institution:** Middlesex University London  
**Academic Year:** 2024-25

---

## Project Overview

An advanced image classification system that recognizes common office items from single images or live camera feed. This project implements and compares **5 deep learning models** (ResNet18 + 4 YOLO variants) to achieve near-perfect classification accuracy on 9 office object categories.

**Achievement:** 99.85% test accuracy with production-ready performance

---

## 🏆 Model Performance Summary

### ResNet18 (Transfer Learning) ✅ TRAINED
**Test Results:**
- **Test Accuracy: 99.85%**
- **Macro F1-Score: 0.9985**
- **Training Time: 115 minutes (25 epochs, 20-core CPU)**
- **Validation Accuracy: 99.88%**

**Perfect Classes (100% F1-Score):**
- Mobile Phone, Notebook, Office Chair, Water Bottle, Keyboard, Monitor

**Near-Perfect:**
- Laptop: 99.60% F1-Score (worst performing, still excellent!)

---

### YOLO Models
**YOLOv8n-cls:** ✅ TRAINED
- **Validation Accuracy: 99.9%**
- **Training Time: ~150 minutes (25 epochs)**
- **Model Size: ~12 MB**

**YOLOv8s-cls, YOLOv11n-cls, YOLOv11s-cls:** 🔄 IN PROGRESS
- Training scheduled on 20-core Intel Core Ultra 7 265K desktop
- Expected completion: Next session

**Ultimate Model Competition:** Planned after all models trained
- Head-to-head comparison on excesses dataset
- Comprehensive performance analysis
- Champion model selection

---

## Classes Recognized (9 Total)

1. Computer Mouse
2. Keyboard
3. Laptop
4. Mobile Phone
5. Monitor
6. Notebook
7. Office Chair
8. Pen
9. Water Bottle

---

## 🔬 Architecture Comparison

### ResNet18 (Transfer Learning)
**Characteristics:**
- Pretrained on ImageNet (1000 classes)
- ~44 MB model size
- Fine-tuned all layers for 9 office classes
- Proven stability and reliability

**Results:**
- ✅ Test Accuracy: **99.85%**
- ✅ 6 classes with perfect 100% F1-score
- ✅ Robust generalization to unseen data

---

### YOLO Family (Modern Lightweight)
**YOLOv8n-cls:**
- ~12 MB (3.7× smaller than ResNet)
- Fast inference
- ✅ Validation: **99.9%** 

**YOLOv8s-cls, YOLOv11n, YOLOv11s:** In training

**Advantages:**
- Smaller model size
- Faster inference
- Optimized for deployment

---

## Dataset

**Version 2.0 Statistics:**
- **Total Images: 22,500** (2,500 per class)
- **Perfectly Balanced:** Equal samples per class
- **Split:** 70% train (15,750), 15% val (3,375), 15% test (3,375)
- **Quality:** High-resolution, diverse, realistic office images

**Data Augmentation (Training):**
- Random crop, horizontal flip, rotation (15°)
- Color jitter (brightness, contrast, saturation ±20%)
- ImageNet normalization

**Sources:** See `data/DATA_SOURCES.md` for attribution

---

## Project Structure
```
office-item-classifier/
├── data/
│   ├── raw/                    # Original images (22.5K, not in Git)
│   ├── processed/              # Train/val/test splits (not in Git)
│   ├── DATASET_CARD.md         # Dataset documentation
│   └── DATA_SOURCES.md         # Source attribution
├── src/
│   ├── organize_dataset.py     # Dataset splitting
│   ├── train.py                # ResNet18 training
│   ├── train_yolo_cls.py       # YOLO training (smart skip logic)
│   ├── evaluate.py             # ResNet evaluation
│   ├── evaluate_yolo_cls.py    # YOLO evaluation
│   ├── model_competition.py    # Ultimate 5-model showdown
│   ├── inference.py            # Single image classification
│   └── camera_inference.py     # Live webcam feed
├── models/
│   ├── best_model.pth          # ResNet18 (99.85% accuracy)
│   ├── final_model.pth         # Final epoch weights
│   └── training_history.json   # Training metrics
├── results/
│   ├── confusion_matrix.png    # ResNet visualization
│   ├── test_metrics.json       # Detailed metrics
│   ├── per_class_metrics.csv   # Per-class performance
│   └── competition/            # Multi-model comparison (upcoming)
├── runs/
│   └── classify/
│       └── yolov8n-cls_train/  # YOLO training outputs
├── excesses/                    # Additional test data (not in Git)
└── requirements.txt            # Dependencies
```

---

## Installation
```bash
# Clone the repository
git clone https://github.com/Shuaibu-oluwatunmise/office-item-classifier
cd office-item-classifier

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- ultralytics (for YOLO)
- OpenCV, scikit-learn, matplotlib, seaborn, pandas

---

## Usage

### Data Preparation
```bash
# 1. Add images to data/raw/[class_name]/ folders
# 2. Organize into train/val/test splits
python src/organize_dataset.py
```

### Training
```bash
# Train ResNet18
python src/train.py

# Train YOLO models (automatically skips already-trained models)
python src/train_yolo_cls.py
```

### Evaluation
```bash
# Evaluate ResNet on test set
python src/evaluate.py

# Evaluate YOLO models
python src/evaluate_yolo_cls.py

# Ultimate model competition (after all models trained)
python src/model_competition.py
```

### Inference
```bash
# Single image classification
python src/inference.py path/to/image.jpg

# Live camera feed
python src/camera_inference.py
```

---

## Training Hardware

**Primary (ResNet Training):**
- CPU: Intel Core Ultra 7 265K (20 cores)
- RAM: 64 GB
- Training Time: 115 minutes (ResNet18, 25 epochs)

**Secondary (Development/Testing):**
- CPU: Intel Core i7-1255U (12 cores)
- RAM: 16 GB
- Used for: Evaluation, inference testing

**Note:** GPU training attempted with RTX 5080 but Blackwell architecture (sm_120) not yet supported by PyTorch 2.6. CPU training proved highly efficient with 20-core processor.

---

## Results & Visualizations

**ResNet18 Evaluation:**
- ✅ Confusion matrix: `results/confusion_matrix.png`
- ✅ Detailed metrics: `results/test_metrics.json`
- ✅ Per-class CSV: `results/per_class_metrics.csv`
- ✅ Classification report: `results/classification_report.txt`

**YOLO Training:**
- ✅ YOLOv8n results: `runs/classify/yolov8n-cls_train/`
- 🔄 Additional models: In progress

**Model Competition:**
- 📅 Scheduled after all 5 models trained
- Will include: Accuracy comparison, confidence analysis, per-class heatmaps, champion selection

---

## Performance Highlights

✅ **99.85% Test Accuracy** (ResNet18)  
✅ **99.88% Validation Accuracy** (ResNet18)  
✅ **6 Perfect Classes** (100% F1-Score)  
✅ **Zero Overfitting** (Val ≈ Test performance)  
✅ **Production Ready** (Robust generalization)  
✅ **Fast Training** (115 min with 20-core CPU)  
✅ **Balanced Dataset** (2,500 images per class)  

---

## Development Progress

**Completed:**
- [x] Dataset collection (22,500 images)
- [x] Data organization and splitting
- [x] ResNet18 training (99.85% test accuracy)
- [x] ResNet18 evaluation and analysis
- [x] YOLOv8n training (99.9% validation)
- [x] Training scripts with smart skip logic
- [x] Model competition framework

**In Progress:**
- [ ] YOLOv8s, YOLOv11n, YOLOv11s training
- [ ] Ultimate 5-model competition
- [ ] Champion model selection

**Upcoming:**
- [ ] Final error analysis
- [ ] Code walkthrough video
- [ ] System overview video
- [ ] Deployment optimization

---

## Key Features

🎯 **Near-Perfect Accuracy:** 99.85% on completely unseen test data  
⚡ **Efficient Training:** Multi-core CPU optimization (16 workers)  
🔄 **Smart Training:** Auto-skip already-trained models  
📊 **Comprehensive Analysis:** Confusion matrices, per-class metrics  
🏆 **Model Competition:** Systematic comparison framework  
🎥 **Real-time Inference:** Live webcam classification  
📦 **Production Ready:** Robust, generalizable models  

---

## Requirements

**Core:**
- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- ultralytics 8.0+

**Supporting:**
- OpenCV (camera inference)
- scikit-learn (metrics)
- matplotlib, seaborn (visualizations)
- pandas (data handling)
- tqdm (progress bars)

See `requirements.txt` for complete list with versions.

---

## Technical Highlights

**Training Optimizations:**
- Multi-worker data loading (16 workers on 20-core CPU)
- Transfer learning with pretrained ImageNet weights
- Data augmentation for robustness
- Learning rate scheduling
- Early stopping based on validation accuracy

**Evaluation Metrics:**
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix analysis
- Confidence score analysis
- Support (samples per class)

---

## License

MIT License - Academic Project

---

## Acknowledgments

- **Middlesex University London** - PDE3802 Module
- **PyTorch & torchvision** - Deep learning framework
- **Ultralytics** - YOLOv8/v11 implementation
- **Dataset Contributors** - See `data/DATA_SOURCES.md`

---

## Citation
```bibtex
@project{office-item-classifier,
  author = {Oluwatunmise Shuaibu Raphael},
  title = {Office Item Classification: Multi-Model Deep Learning Comparison},
  year = {2024-2025},
  institution = {Middlesex University London},
  module = {PDE3802 - AI in Robotics}
}
```

---

## Contact

**Student:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Module:** PDE3802 - AI in Robotics  
**Institution:** Middlesex University London

---

*Last Updated: October 21, 2024*  
*Current Status: ResNet18 complete (99.85%), YOLO training in progress*