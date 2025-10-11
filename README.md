# Office Item Classifier

**Module:** PDE3802 - AI in Robotics  
**Assessment:** Part B - Office-Goods Classification Code  
**Author:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Institution:** Middlesex University London  
**Academic Year:** 2025-26

---

## Project Overview

An image classification system that recognizes common office items from single images or live camera feed. This project uses deep learning to classify 11 office object categories, with systematic comparison of two architectures (ResNet18 and YOLOv8) for optimal performance.

**Focus:** Bridging the domain gap between clean training data and messy real-world deployment conditions.

---

## 🔄 Model Versions

### Version 2.0 (Current - In Progress)

**Status:** Data collection and dual architecture training phase

**Strategy:**
- **Dual Architecture Approach:** Training both ResNet18 and YOLOv8n-cls on identical dataset
- **Improved Data Quality:** Focus on diverse, realistic office images
- **Systematic Comparison:** Evaluate both models on same test set
- **Goal:** 85-90% accuracy in real-world camera conditions

**Why Version 2.0:**
Version 1.0 achieved 96.37% test accuracy but struggled with real-world camera performance due to domain gap between clean training images (isolated objects, plain backgrounds) and messy deployment conditions (complex backgrounds, varied lighting, natural angles).

**Improvements:**
- Higher quality, more diverse training data
- Multiple viewing angles and lighting conditions
- Real office environment representation
- Better generalization to deployment scenarios

---

### Version 1.0 (Archived)

**Performance:**
- Test Accuracy: 96.37%
- Macro F1-Score: 0.9528
- Architecture: ResNet18 only
- Dataset: 13,616 images from Roboflow Universe

**Limitation Identified:**
Poor real-world camera performance (~60-70% accuracy) due to clean, isolated training images not representing deployment conditions.

**📁 Complete Version 1.0 Archive:**
All v1.0 files (models, results, documentation, analysis) available in `/legacy` folder:
- `legacy/models_v1/` - Trained model weights
- `legacy/results_v1/` - Evaluation metrics and confusion matrix
- `legacy/docs_v1/` - Dataset documentation and error analysis
- `legacy/main_README_v1.md` - Original README
- `legacy/src_README_v1.md` - Original script documentation

---

## 🔬 Architecture Comparison

Version 2.0 implements **two architectures** trained on identical data for systematic comparison:

### ResNet18 (Transfer Learning)
**Characteristics:**
- Pretrained on ImageNet (1000 classes)
- 44.8 MB model size
- Fine-tuned all layers for 11 office classes
- Proven stability and reliability

**Expected:**
- High accuracy on clean test images
- Moderate inference speed (~50-70ms)
- Robust feature extraction

---

### YOLOv8n-cls (Modern Lightweight)
**Characteristics:**
- Optimized for real-time classification
- ~12 MB model size (3.7× smaller than ResNet)
- Fast inference (~30-40ms, 1.5-2× faster)
- Efficient architecture

**Expected:**
- Competitive accuracy
- Faster inference
- Better for deployment/mobile

---

**Comparison Methodology:**
- Identical dataset (same train/val/test split)
- Same training parameters (epochs, batch size)
- Same evaluation metrics (accuracy, F1, confusion matrix)
- Tested on same hardware (Intel i7-1255U CPU)

**Results:** TBD - See `notes/MODEL_COMPARISON.md` after training completes

---

## Classes Recognized

1. Computer Mouse
2. Keyboard
3. Laptop
4. Mobile Phone
5. Mug
6. Notebook
7. Office Bin
8. Office Chair
9. Pen
10. Stapler
11. Water Bottle

---

## Project Structure

```
office-item-classifier/
├── data/
│   ├── raw/                    # Training images (not in Git)
│   ├── processed/              # Train/val/test splits (not in Git)
│   ├── dataset_card.md         # Dataset documentation (v2.0)
│   ├── DATA_SOURCES.md         # Source attribution (v2.0)
│   └── yolo_data.yaml          # YOLO configuration
├── src/
│   ├── organize_dataset.py     # Dataset splitting
│   ├── train.py                # ResNet18 training
│   ├── train_yolo_cls.py       # YOLOv8 training
│   ├── evaluate.py             # ResNet evaluation
│   ├── evaluate_yolo_cls.py    # YOLO evaluation
│   ├── inference.py            # Single image classification
│   ├── camera_inference.py     # Live webcam feed
│   └── archive/                # Legacy exploration scripts
├── models/
│   └── (trained models saved here)
├── results/
│   ├── (ResNet results)
│   └── yolo/ (YOLO results)
├── runs/
│   └── classify/ (YOLO training outputs)
├── legacy/
│   ├── models_v1/              # Version 1.0 models
│   ├── results_v1/             # Version 1.0 results
│   ├── docs_v1/                # Version 1.0 documentation
│   └── README files            # Version 1.0 documentation
├── notes/
│   └── MODEL_COMPARISON.md     # Architecture comparison
└── docs/
    └── ERROR_ANALYSIS.md       # Current version analysis (TBD)
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
# Train ResNet18 (~11 hours on CPU)
python src/train.py

# Train YOLOv8n-cls (~8-9 hours on CPU)
python src/train_yolo_cls.py
```

### Evaluation

```bash
# Evaluate ResNet
python src/evaluate.py

# Evaluate YOLO
python src/evaluate_yolo_cls.py
```

### Inference

```bash
# Single image classification (ResNet)
python src/inference.py path/to/image.jpg

# Live camera feed (ResNet)
python src/camera_inference.py
```

---

## Dataset

**Version 2.0 (In Progress):**
- **Focus:** High-quality, diverse, realistic office images
- **Sources:** TBD - See `data/DATA_SOURCES.md`
- **Size:** TBD
- **Split:** 70% train, 15% validation, 15% test

**Version 1.0 (Archived):**
- **Source:** Roboflow Universe (11 projects)
- **Size:** 13,616 images
- **Details:** See `legacy/docs_v1/dataset_card_v1.md`

**Data Augmentation (Training):**
- Random crop, horizontal flip, rotation
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

---

## Development Progress

**Completed:**
- [x] Version 1.0 training and evaluation
- [x] Version 1.0 error analysis
- [x] Dual architecture implementation
- [x] Training/evaluation pipelines for both models
- [x] Inference scripts (file and camera)

**In Progress:**
- [ ] Version 2.0 data collection
- [ ] Dual architecture training
- [ ] Systematic performance comparison
- [ ] Final model selection

**Upcoming:**
- [ ] Error analysis (v2.0)
- [ ] Code walkthrough video
- [ ] System overview video

---

## Performance Targets

**Version 2.0 Goals:**
- **Test Accuracy:** 94-96%
- **Real-world Camera:** 85-90% (significant improvement over v1.0)
- **Inference Speed:** <50ms per image
- **Model Size:** <50 MB

**Success Criteria:**
- Consistent performance across test set and real-world conditions
- Minimal domain gap between evaluation and deployment
- Robust to varied backgrounds, lighting, and angles

---

## Requirements

**Core:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- ultralytics

**Supporting:**
- OpenCV (camera inference)
- scikit-learn (metrics)
- matplotlib, seaborn (visualizations)
- pandas (data handling)

See `requirements.txt` for complete list.

---

## Hardware

**Training Hardware:**
- CPU: Intel Core i7-1255U (12th Gen)
- RAM: 16 GB
- GPU: Intel Iris Xe Graphics (integrated)

**Training Time:**
- ResNet18: ~11 hours (25 epochs)
- YOLOv8n-cls: ~8-9 hours (25 epochs)

---

## License

MIT License - Academic Project

---

## Acknowledgments

- **Middlesex University London** - PDE3802 Module
- **PyTorch** - Deep learning framework
- **Ultralytics** - YOLO implementation
- **Roboflow Universe** - Dataset sources (v1.0)

---

## Version History

- **v1.0** (Oct 10, 2025): ResNet18, 96.37% test accuracy → Archived in `/legacy`
- **v2.0** (In Progress): Dual architecture, improved data, real-world focus

---

## Contact

**Student:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Module:** PDE3802 - AI in Robotics  
**Institution:** Middlesex University London

---

*Last Updated: October 10, 2025*  
*Version 2.0 - Dual architecture approach with improved dataset for real-world deployment*