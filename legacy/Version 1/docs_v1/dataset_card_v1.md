# Dataset Card: Office Items Classification Dataset

## Dataset Description

**Version:** 2.0 (Complete Collection)  
**Date Created:** October 9-10, 2025  
**Last Updated:** October 10, 2025  
**Purpose:** Training and evaluation of office item classification model for PDE3802 assessment  
**Total Images:** 13,616  
**Total Classes:** 11

## Dataset Summary

This dataset contains images of 11 common office items collected from Roboflow Universe for training an autonomous office organizing robot's perception system. The dataset is carefully balanced and quality-controlled for effective model training.

### Classes (11 total)

1. **Computer Mouse** - Computer mice (wired/wireless)
2. **Keyboard** - Computer keyboards (various types)
3. **Stapler** - Office staplers
4. **Laptop** - Laptop computers
5. **Office Chair** - Desk chairs, office seating
6. **Mug** - Coffee mugs, tea cups, drinking cups
7. **Pen** - Pens and writing instruments
8. **Notebook** - Paper notebooks, journals
9. **Mobile Phone** - Smartphones and mobile devices
10. **Office Bin** - Waste bins, recycling bins
11. **Water Bottle** - Reusable water bottles

## Data Sources

### All Datasets from Roboflow Universe

All 11 classes were sourced from publicly available projects on Roboflow Universe. Each dataset was manually reviewed before download to ensure quality and relevance.

**Download Method:**
- Format: COCO JSON
- Splits: Combined train/valid/test into single class folders
- Manual download and extraction
- Quality pre-screening before download

For detailed attribution of each dataset, see `DATA_SOURCES.md`.

### Key Source Information

| Class | Project | Images |
|-------|---------|--------|
| Laptop | laptop-detection | 1,547 |
| Mobile Phone | cell-phone-detect | 1,670 |
| Office Bin | bin-detection-test2 | 1,668 |
| Notebook | notebook-eiabn | 1,500* |
| Water Bottle | water-bottle-eovbe | 1,356 |
| Stapler | stapler-rdudj | 1,354 |
| Pen | pen-detection-lexd8 | 915 |
| Keyboard | all-keyboard | 811 |
| Mug | cup-detection-w8kfb | 794 |
| Office Chair | chair-bgkdb | 777 |
| Computer Mouse | office-mouse | 724 |

*Reduced from 2,591 through random sampling for balance

## Current Dataset Statistics (Raw Data)

### Class Distribution

| Class          | Images | Percentage | Status      |
|----------------|--------|------------|-------------|
| Laptop         | 1,547  | 11.4%      | Good        |
| Mobile Phone   | 1,670  | 12.3%      | Good        |
| Office Bin     | 1,668  | 12.3%      | Good        |
| Notebook       | 1,500  | 11.0%      | Balanced    |
| Water Bottle   | 1,356  | 10.0%      | Good        |
| Stapler        | 1,354  | 9.9%       | Good        |
| Pen            | 915    | 6.7%       | Balanced    |
| Keyboard       | 811    | 6.0%       | Balanced    |
| Mug            | 794    | 5.8%       | Balanced    |
| Office Chair   | 777    | 5.7%       | Balanced    |
| Computer Mouse | 724    | 5.3%       | Balanced    |
| **TOTAL**      | **13,616** | **100%** | **Complete** |

### Balance Analysis

- **Most Images:** Mobile Phone (1,670) and Office Bin (1,668)
- **Fewest Images:** Computer Mouse (724)
- **Ratio:** 2.3:1 (largest to smallest class)
- **Assessment:** Well-balanced dataset suitable for classification

All classes have >700 images, which is sufficient for training robust deep learning models.

## Train/Validation/Test Split

**Status:** Not yet split. All images currently in `data/raw/[class_name]/`

**Planned Split Ratios:**
- **Training:** 70% (~9,531 images)
- **Validation:** 15% (~2,042 images)
- **Testing:** 15% (~2,043 images)

Split will be performed using `src/organize_dataset.py` with stratified sampling to maintain class proportions.

## Data Collection Methodology

### Phase 1: Source Identification (October 9, 2025)
1. Searched Roboflow Universe for each target office item class
2. Evaluated available datasets for:
   - Image quality
   - Variety (angles, backgrounds, lighting)
   - Relevance to office environment
   - Sufficient quantity (target: 500+ per class)
3. Selected best available project for each class

### Phase 2: Download and Organization (October 9-10, 2025)
1. Downloaded each dataset in COCO JSON format from Roboflow
2. Extracted all splits (train, valid, test) from downloaded archives
3. Combined all images from all splits into unified class folders
4. Copied to `data/raw/[class_name]/` directory structure

### Phase 3: Quality Control and Balancing (October 10, 2025)
1. Verified image counts for all classes
2. Identified notebook class as overrepresented (2,591 images)
3. Randomly sampled 1,500 images from notebook to improve balance
4. Final dataset: 13,616 images across 11 classes

### Phase 4: Future Processing (Planned)
1. Manual review and deletion of any poor-quality images
2. Train/validation/test split using stratified sampling
3. Image preprocessing (resize to 224x224, normalization)
4. Data augmentation during training (rotation, flip, brightness, etc.)

## Data Format

- **Image Formats:** JPEG, PNG
- **Current Storage:** `data/raw/[class_name]/[image_name].[ext]`
- **Target Size:** 224×224 pixels (for model input)
- **Normalization:** ImageNet statistics (planned during training)
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

## Quality Considerations

### Strengths
- **High Volume:** 13,616 total images
- **Well-Balanced:** Largest class only 2.3× smallest class
- **Diverse Sources:** 11 different Roboflow projects
- **Quality Pre-screened:** Manual review before download
- **Real-World:** Images from actual office environments

### Limitations
- **Domain Variety:** All images from Roboflow (primarily web-sourced)
- **No Custom Photos:** Missing real photos from target office environment
- **Combined Splits:** Original train/valid/test boundaries not preserved
- **Class Overlap:** Some classes may have visual similarity (e.g., mug vs. water bottle)

### Future Improvements
- Add custom photos from actual office environment
- Augment classes with fewer images (mouse, chair, mug)
- Include more diverse backgrounds and lighting conditions
- Add images of objects in different states (e.g., open/closed notebooks)

## Ethical Considerations

- **Privacy:** No personal or sensitive data in images
- **Attribution:** All sources properly cited in DATA_SOURCES.md
- **Licensing:** Each dataset follows its original Roboflow Universe license
- **Usage:** Data used solely for educational/academic purposes
- **Redistribution:** Images not redistributed; only code and methodology shared

## Known Issues

1. **Water Bottle Dataset:** Only contained train split (no valid/test)
2. **Class Imbalance:** Some variation in class sizes (724 to 1,670 images)
3. **Label Ambiguity:** "Pen" class may include pencils (depends on source)
4. **Object Similarity:** Mug and water bottle may be visually similar in some images

## Usage Guidelines

### For Training
```python
# Recommended data augmentation
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### For Evaluation
```python
# No augmentation for validation/test
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

## Evaluation Metrics

Planned metrics for model evaluation:
- **Accuracy:** Overall classification accuracy
- **Macro F1-Score:** Average F1 across all classes
- **Confusion Matrix:** Class-wise performance analysis
- **Per-Class Precision/Recall:** Identify weak classes
- **Top-3 Accuracy:** Proportion where true class is in top 3 predictions

## Dataset Versions

- **v1.0** (Oct 9, 2025): Initial collection from Kaggle, 1,247 images, 5 classes
- **v2.0** (Oct 10, 2025): Complete Roboflow collection, 13,616 images, 11 classes

## File Structure

```
data/
├── raw/                          # All images, organized by class
│   ├── computer_mouse/           # 724 images
│   ├── keyboard/                 # 811 images
│   ├── stapler/                  # 1,354 images
│   ├── laptop/                   # 1,547 images
│   ├── office_chair/             # 777 images
│   ├── mug/                      # 794 images
│   ├── pen/                      # 915 images
│   ├── notebook/                 # 1,500 images
│   ├── mobile_phone/             # 1,670 images
│   ├── office_bin/               # 1,668 images
│   └── water_bottle/             # 1,356 images
├── processed/                    # Will contain train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
└── dataset_card.md               # This file
```

## Citation

```bibtex
@misc{office_items_dataset_2025,
  title={Office Items Classification Dataset},
  author={Raphael, Oluwatunmise Shuaibu},
  year={2025},
  institution={Middlesex University London},
  note={PDE3802 Assessment - 13,616 images across 11 office item classes}
}
```

## Acknowledgments

This dataset aggregates publicly available data from Roboflow Universe. We thank all dataset contributors who made their work publicly available. Full attribution in DATA_SOURCES.md.

## Contact & Support

- **Student:** Oluwatunmise Shuaibu Raphael
- **Student ID:** M00960413
- **Module:** PDE3802 - AI in Robotics
- **Institution:** Middlesex University London
- **Academic Year:** 2025-26

---

*This dataset card follows ML best practices for dataset documentation and transparency.*  
*Last Updated: October 10, 2025*