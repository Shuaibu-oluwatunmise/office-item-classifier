# Legacy Versions - Project Evolution

This document chronicles the experimental and iterative development phases that led to the current Deployment system. The legacy folder contains two major versions representing learning experiments and architectural explorations.

---

## Overview

The legacy folder preserves two development phases:
- **Version 1:** Initial ResNet18 prototype (October 2025)
- **Version 2:** Multi-model comparison and dataset enhancement (October 2025)

These versions are archived to document the learning process, design decisions, and evolution toward the current Deployment system.

---

## Version 1: Initial Prototype

### 📅 Timeline
**October 9-10, 2025**

### 🎯 Objectives
- Prove feasibility of office item classification
- Establish baseline performance
- Test transfer learning approach

### 🏗️ Architecture
**Model:** ResNet18 (Transfer Learning)
- Base: ImageNet pretrained weights
- Modified: Final layer for 11 classes
- Framework: PyTorch
- Training: CPU-based (Intel i7-1255U, 10 hours 48 minutes)

### 📊 Dataset
**Source:** Roboflow Universe (11 different projects)
- **Total Images:** 13,616
- **Classes:** 11 (Computer Mouse, Keyboard, Laptop, Mobile Phone, Mug, Notebook, Office Bin, Office Chair, Pen, Stapler, Water Bottle)
- **Split:** 70% train (10,005), 15% val (2,473), 15% test (2,482)
- **Characteristics:** Clean, isolated objects on plain backgrounds

### 📈 Results
**Test Performance:**
- **Accuracy:** 96.37%
- **Macro F1-Score:** 0.9528
- **Best Class:** Office Chair (99.29% F1)
- **Worst Class:** Computer Mouse (85.57% F1)

**Training Performance:**
- **Best Val Accuracy:** 97.45% (Epoch 22)
- **Final Val Accuracy:** 95.99% (Epoch 25)
- **Minimal Overfitting:** 0.05% gap

### ❌ Critical Problem: Domain Gap
**Real-World Performance:** ~60-70% accuracy

**Root Cause:**
- Training data: Clean, isolated objects, plain backgrounds, perfect lighting
- Real-world: Cluttered desks, complex backgrounds, varied lighting, occlusions
- **Domain gap** between training and deployment was too large

**Key Confusions:**
- Mouse → Stapler (10% of mice misclassified)
- Similar compact form factors caused confusion
- Viewing angle dependencies

### 📚 Key Learnings
1. ✅ Transfer learning works well for office items
2. ✅ ResNet18 is efficient and reliable
3. ❌ Clean web-scraped data doesn't generalize to messy environments
4. ❌ Need diverse backgrounds, lighting, and viewing angles
5. ❌ Isolated object images fail in cluttered real-world scenarios

### 📁 Files Preserved
```
legacy/Version 1/
├── main_README_v1.md
├── src_README_v1.md
├── docs_v1/
│   ├── dataset_card_v1.md
│   ├── DATA_SOURCES_v1.md
│   └── ERROR_ANALYSIS.md
├── models_v1/
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.json
└── results_v1/
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    ├── per_class_metrics.csv
    └── test_metrics.json
```

---

## Version 2: Multi-Model Comparison & Dataset Enhancement

### 📅 Timeline
**October 10-23, 2025**

### 🎯 Objectives
- Bridge the domain gap from V1
- Compare multiple architectures
- Achieve real-world deployment readiness
- Find optimal model for Deployment

### 🏗️ Architecture
**5-Model Competition:**

1. **ResNet18** (Transfer Learning)
   - PyTorch implementation
   - ImageNet pretrained
   - 44 MB model size

2. **YOLOv8n-cls** (Nano)
   - Ultralytics framework
   - ~12 MB (3.7× smaller than ResNet)
   - Fast inference

3. **YOLOv8s-cls** (Small)
   - Larger than nano but still lightweight
   - Better accuracy/size tradeoff

4. **YOLOv11n-cls** (Nano)
   - Latest YOLO version
   - Improved architecture

5. **YOLOv11s-cls** (Small)
   - Newest small variant
   - Enhanced features

### 📊 Dataset Improvements
**Source:** Enhanced collection strategy
- **Total Images:** 22,500 (balanced)
- **Images per Class:** 2,500 (perfectly balanced)
- **Classes:** 9 (reduced from 11 - focused on most common items)
  - Removed: Office Bin, Mobile Phone
  - Kept: Computer Mouse, Keyboard, Laptop, Monitor, Mug, Notebook, Office Chair, Pen, Water Bottle

**Quality Enhancements:**
- ✅ Diverse backgrounds (desks, shelves, in-use)
- ✅ Multiple viewing angles
- ✅ Various lighting conditions
- ✅ Real office contexts
- ✅ Perfect class balance

**Split:** 70% train (15,750), 15% val (3,375), 15% test (3,375)

### 📈 Results

#### Individual Model Performance (Validation)

| Model | Accuracy | Model Size | Training Time |
|-------|----------|------------|---------------|
| ResNet18 | 99.88% | 44 MB | 115 min |
| YOLOv8n | 99.90% | 12 MB | ~150 min |
| **YOLOv8s** | **99.90%** | ~25 MB | ~180 min |
| YOLOv11n | 99.90% | 12 MB | ~150 min |
| YOLOv11s | 99.90% | ~25 MB | ~180 min |

#### Ultimate Competition (12,091 test images - "excesses" folder)

🏆 **Final Rankings:**

1. 🥇 **YOLOv8s** - 99.99% accuracy
2. 🥈 **YOLOv8n** - 99.98% accuracy
3. 🥉 **YOLOv11n** - 99.98% accuracy
4. **YOLOv11s** - 99.97% accuracy
5. **ResNet18** - 99.80% accuracy

**Confidence Analysis:**
- All YOLO models: >99.97% average confidence
- ResNet18: 99.40% average confidence
- All models showed robust performance

### 🎯 Key Insight: Model Selection

**Competition Winner:** YOLOv8s (99.99% accuracy)

**Recommended for Deployment:** YOLOv8n (99.98% accuracy)

**Rationale:**
- Negligible accuracy difference: 0.01% (1 image in 10,000)
- YOLOv8n significantly faster inference
- 2× smaller size (12 MB vs 25 MB)
- Better suited for edge devices
- Speed and size critical for deployment

### ✅ Success: Domain Gap Bridged
**Real-World Performance:** Estimated 85-90% accuracy (significant improvement from V1's 60-70%)

**Key Improvements:**
- Diverse training data eliminated domain gap
- Realistic office environments during training
- Multiple viewing angles improved robustness
- Lighting variation enhanced generalization

### 🔧 Training Infrastructure
**Hardware Upgrade:** Intel Core Ultra 7 265K (20 cores, 64GB RAM)
- ResNet18: 115 minutes (25 epochs)
- YOLO models: ~150-180 minutes each
- Multi-worker optimization (16 workers)
- Smart skip logic (avoid retraining)

### 📚 Key Learnings
1. ✅ Dataset quality > model architecture for real-world performance
2. ✅ YOLO models highly competitive with traditional CNNs
3. ✅ Balanced datasets crucial for equal class performance
4. ✅ Diverse backgrounds/lighting essential for deployment
5. ✅ Smaller models sufficient when data is high-quality
6. ✅ Speed/size tradeoffs matter for deployment
7. ✅ Model competition framework valuable for systematic comparison

### 📁 Files Preserved
```
legacy/Version 2/
├── README.md
├── DATASET_CARD.md
├── DATA_SOURCES.md
├── models/
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.json
├── results/
│   ├── confusion_matrix.png
│   ├── test_metrics.json
│   ├── per_class_metrics.csv
│   ├── competition/
│   │   ├── competition_report.txt
│   │   ├── detailed_results.json
│   │   ├── leaderboard.csv
│   │   └── model_competition_results.png
│   └── yolo/
│       └── yolo_comparison.json
├── runs/classify/
│   ├── yolov8n-cls_train/
│   ├── yolov8s-cls_train/
│   ├── yolov11n-cls_train/
│   └── yolov11s-cls_train/
└── src/
    ├── train.py
    ├── train_yolo_cls.py
    ├── evaluate.py
    ├── evaluate_yolo_cls.py
    ├── model_competition.py
    └── README.md
```

---

## Archive Folder

Contains experimental scripts from early development:
- `download_datasets.py` - Kaggle dataset downloader
- `download_kaggle_datasets.py` - Kaggle API integration
- `inspect_downloads.py` - Dataset structure explorer
- `smart_organize.py` - Early organization attempts
- `ML_CONCEPTS_EXPLAINED.md` - Learning notes

**Status:** Historical reference only, not used in Deployment

---

## Evolution Summary

### Dataset Evolution

| Version | Images | Classes | Balance | Diversity | Real-World Ready |
|---------|--------|---------|---------|-----------|------------------|
| V1 | 13,616 | 11 | Imbalanced | Low | ❌ No |
| V2 | 22,500 | 9 | Perfect | High | ✅ Yes |

### Model Evolution

| Version | Approach | Best Accuracy | Real-World | Deployment |
|---------|----------|---------------|------------|------------|
| V1 | ResNet18 only | 96.37% test | ❌ 60-70% | ❌ No |
| V2 | 5-model competition | 99.99% test | ✅ 85-90% | ⚠️ Models ready, pipeline incomplete |

### Infrastructure Evolution

| Version | Hardware | Training Time | GPU | Pipeline |
|---------|----------|---------------|-----|----------|
| V1 | i7-1255U (12 cores) | 10h 48m | ❌ None | Manual |
| V2 | Core Ultra 7 265K (20 cores) | 115-180m | ❌ None | Semi-automated |

---

## Key Takeaways

### What Worked
1. ✅ Transfer learning is highly effective for office items
2. ✅ YOLOv8n offers best accuracy/speed/size tradeoff
3. ✅ Dataset quality > model complexity
4. ✅ Systematic model comparison prevents premature optimization
5. ✅ Balanced datasets improve per-class performance

### What Didn't Work
1. ❌ Clean web-scraped data for real-world deployment
2. ❌ Imbalanced datasets (even slight imbalance impacts performance)
3. ❌ Plain backgrounds only (creates domain gap)
4. ❌ Ignoring deployment constraints (size, speed)
5. ❌ CPU-only training at scale (acceptable but slow)

### Critical Lessons
1. **Test in deployment conditions early** - V1's 96% test accuracy was meaningless in real-world (60-70%)
2. **Invest in dataset quality** - Hours spent on diverse data > days spent on model tuning
3. **Benchmark multiple models** - YOLOv8s won competition but YOLOv8n better for deployment
4. **Document everything** - This file ensures lessons aren't forgotten
5. **Iterate quickly** - V2 built on V1's failures within 2 weeks

---

## Path Forward

The learnings from V1 and V2 informed the current Deployment system:
- **From V1:** Importance of dataset diversity and real-world testing
- **From V2:** Systematic model comparison and deployment considerations

These archived versions remain valuable for:
- Understanding design decisions
- Avoiding repeated mistakes
- Training new team members
- Academic documentation

---

**Author:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Module:** PDE3802 - AI in Robotics  
**Institution:** Middlesex University London  

*Last Updated: November 7, 2025*  
*Status: V1 & V2 archived, current system in Deployment*