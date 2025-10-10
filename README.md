# Office Item Classifier

**Module:** PDE3802 - AI in Robotics  
**Assessment:** Part B - Office-Goods Classification Code  
**Author:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413

## Project Overview

An image classification system that recognizes common office items from single images or live camera feed. This project uses deep learning (PyTorch with ResNet18 transfer learning) to classify 11 office object categories with **97.45% validation accuracy**.

## 🎯 Training Results

**Model Performance:**
- ✅ **Best Validation Accuracy:** 97.45% (Epoch 22) 🏆
- ✅ **Final Validation Accuracy:** 95.99% (Epoch 25)
- ✅ **Training Accuracy:** 97.50%
- ✅ **Minimal Overfitting:** Only 0.05% gap between train and validation

**Training Details:**
- **Duration:** 10 hours 48 minutes (648 minutes)
- **Hardware:** Intel Core i7-1255U (CPU), 16GB RAM, Intel Iris Xe Graphics
- **Model:** ResNet18 (pretrained on ImageNet, fine-tuned)
- **Optimizer:** Adam (Learning Rate: 0.001)
- **Batch Size:** 32
- **Epochs:** 25

**Dataset:**
- **Total Images:** 13,616
- **Training:** 10,005 images (70%)
- **Validation:** 2,473 images (15%)
- **Testing:** ~2,043 images (15%)
- **Classes:** 11

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

## Project Structure

```
office-item-classifier/
├── data/
│   ├── raw/                    # Original images (not in Git - 13,616 images)
│   ├── processed/              # Train/val/test splits (not in Git)
│   ├── dataset_card.md         # Complete dataset documentation
│   └── DATA_SOURCES.md         # Full attribution
├── src/
│   ├── train.py                # Training pipeline (ResNet18)
│   ├── organize_dataset.py     # Dataset splitting script
│   ├── evaluate.py             # [IN PROGRESS] Test set evaluation
│   ├── inference.py            # [TODO] Single image classification
│   └── camera_inference.py     # [TODO] Live camera feed
├── models/
│   ├── best_model.pth          # Best model (97.45% val acc)
│   ├── final_model.pth         # Final epoch model
│   └── training_history.json   # All training metrics
├── results/
│   └── (evaluation results will go here)
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Shuaibu-oluwatunmise/office-item-classifier
cd office-item-classifier

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training (Already Complete ✅)

```bash
python src/train.py
```

**Output:**
- `models/best_model.pth` - Best validation accuracy model
- `models/final_model.pth` - Final epoch model
- `models/training_history.json` - Training metrics

### Evaluation (Next Step)

```bash
python src/evaluate.py
```

**Will generate:**
- Test accuracy, macro F1-score
- Confusion matrix
- Per-class performance metrics

### Inference (Coming Soon)

```bash
# Single image
python src/inference.py path/to/image.jpg

# Live camera feed
python src/camera_inference.py
```

## Dataset

**Source:** Roboflow Universe (11 different projects)  
**Total Images:** 13,616 across 11 balanced classes  
**Format:** JPEG/PNG, resized to 224×224 for training  

**Data Augmentation:**
- Random crop (224×224)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation)
- Random rotation (±15°)
- ImageNet normalization

**Note:** Dataset images are excluded from Git (too large). See `data/dataset_card.md` for complete documentation and sources.

## Model Architecture

**Base Model:** ResNet18 (pretrained on ImageNet)  
**Modifications:**
- Replaced final fully connected layer for 11 classes
- Fine-tuned all layers (not frozen)
- Cross-entropy loss
- Adam optimizer

**Why ResNet18?**
- Proven architecture for image classification
- Lightweight (44.8 MB) - suitable for deployment
- Pretrained weights provide strong feature extraction
- Fast training even on CPU

## Test Results Summary

**Performance Metrics:**
- **Test Accuracy:** 96.37% (2,482 test images)
- **Macro F1-Score:** 0.9528
- **Best Performing Class:** Office Chair (99.29% F1)
- **Most Challenging Class:** Computer Mouse (85.57% F1)

**Key Findings:**
- Minimal overfitting: Only 1.08% gap between validation (97.45%) and test (96.37%)
- 8 out of 11 classes achieve >95% F1-score
- Main confusion: Computer mouse → stapler (10% of mice)
- Root cause: Similar compact handheld form factors and ergonomic shapes

**Detailed Analysis:** See `docs/ERROR_ANALYSIS.md` for comprehensive error analysis, confusion patterns, and improvement recommendations.

## Development Progress

- [x] Project structure setup
- [x] Dataset collection (13,616 images from Roboflow)
- [x] Dataset organization (70/15/15 split)
- [x] Data preprocessing and augmentation
- [x] Model training (97.45% validation accuracy)
- [x] Test set evaluation (96.37% accuracy, 0.9528 macro F1)
- [x] Error analysis and confusion matrix
- [ ] Inference scripts (file and camera)
- [ ] Documentation and video walkthrough

## Requirements

**Key Dependencies:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- OpenCV (for camera inference)
- scikit-learn
- matplotlib
- tqdm

See `requirements.txt` for complete list.

## Next Steps

1. Evaluate model on test set
2. Generate confusion matrix and metrics
3. Implement inference scripts
4. Analyze common classification errors
5. Create code walkthrough video

## License

MIT License - Academic Project

## Acknowledgments

- **Middlesex University London** - PDE3802 Module
- **Roboflow Universe** - Dataset sources (see DATA_SOURCES.md)
- **PyTorch** - Deep learning framework

---

*Last Updated: October 10, 2025*  
*Training completed with 97.45% validation accuracy*