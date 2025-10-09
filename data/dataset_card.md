# Dataset Card: Office Items Classification Dataset

## Dataset Description

**Version:** 1.0 (Initial Download Phase)  
**Date Created:** October 9, 2025  
**Last Updated:** October 9, 2025  
**Purpose:** Training and evaluation of office item classification model for PDE3802 assessment

## Dataset Summary

This dataset contains images of 10 common office items collected for the purpose of training an autonomous office organizing robot's perception system. Currently in the data collection phase.

### Classes (10 total)

1. **Mug** - Coffee mugs, tea cups, drinking cups
2. **Water Bottle** - Reusable water bottles, plastic bottles
3. **Mobile Phone** - Smartphones
4. **Keyboard** - Computer keyboards (wired/wireless)
5. **Computer Mouse** - Computer mice (wired/wireless)
6. **Stapler** - Office staplers
7. **Pen/Pencil** - Writing instruments
8. **Notebook** - Paper notebooks, journals
9. **Office Chair** - Desk chairs, office seating
10. **Office Bin** - Waste bins, recycling bins

## Data Sources

### Downloaded Datasets (Kaggle)

All datasets downloaded on October 9, 2025 using Kaggle API.

1. **Cup and Mug Dataset**
   - Source: malikusman1221/cup-mug-dataset
   - License: See Kaggle dataset page
   - Contributed: 1044 images to 'mug' class

2. **Mouse and Keyboard Dataset**
   - Source: vshalsgh/mouse-and-keyboard-dataset
   - License: See Kaggle dataset page
   - Contributed: 31 images to 'computer_mouse', 79 to 'keyboard'

3. **Bottles and Cups Dataset**
   - Source: dataclusterlabs/bottles-and-cups-dataset
   - License: See Kaggle dataset page
   - Contributed: 76 images to 'water_bottle', 14 to 'mug'

4. **Electronics Dataset**
   - Source: dataclusterlabs/electronics-mouse-keyboard-image-dataset
   - License: See Kaggle dataset page
   - Contributed: 3 images to 'mobile_phone'

For detailed attribution, see `DATA_SOURCES.md`

### Custom Dataset (Self-captured)
- **Capture Date:** To be added
- **Equipment:** To be documented
- **Number of images per class:** To be added

## Current Dataset Statistics (Raw Data - Pre-split)

| Class          | Images | Status       | Notes                    |
|----------------|--------|--------------|--------------------------|
| Mug            | 1058   | Sufficient   | Well-represented         |
| Keyboard       | 79     | Sufficient   | Good variety             |
| Water Bottle   | 76     | Sufficient   | Good variety             |
| Computer Mouse | 31     | Borderline   | Need 20+ more            |
| Mobile Phone   | 3      | Insufficient | Need 50+ more            |
| Stapler        | 0      | Missing      | Data collection needed   |
| Pen/Pencil     | 0      | Missing      | Data collection needed   |
| Notebook       | 0      | Missing      | Data collection needed   |
| Office Chair   | 0      | Missing      | Data collection needed   |
| Office Bin     | 0      | Missing      | Data collection needed   |
| **TOTAL**      | **1247** | **Partial** | **5/10 classes ready** |

## Train/Validation/Test Split

**Status:** Not yet split. Raw images currently in `data/raw/`

**Planned Split Ratios:**
- Training: 70%
- Validation: 15%
- Testing: 15%

Split will be performed using `organize_dataset.py` script after all data collection is complete.

## Data Collection Methodology

### Phase 1: Initial Download (Complete)
1. Identified relevant Kaggle datasets for target classes
2. Downloaded using Kaggle API (automated, reproducible)
3. Organized images by parsing annotations where needed (XML, YOLO format)
4. Copied to class-specific folders in `data/raw/`

### Phase 2: Additional Downloads (In Progress)
- Finding datasets for missing classes
- Supplementing insufficient classes

### Phase 3: Custom Capture (Planned)
- Multiple angles per object
- Various lighting conditions
- Different backgrounds
- Real office environment

## Data Processing

- **Image formats:** JPEG, PNG
- **Target size:** 224x224 pixels (for model input)
- **Normalization:** ImageNet statistics (planned)
- **Augmentation:** To be applied during training
- **Preprocessing scripts:** `organize_dataset.py`, `smart_organize.py`

## Quality Control

- Automated organization with class mapping
- XML annotation parsing for object detection datasets
- Manual verification planned before training
- Removal of corrupted/invalid images (if found)

## Ethical Considerations

- No personal or sensitive data in images
- Public spaces or stock images only
- Proper attribution of all source datasets
- Used for educational purposes only (academic assessment)

## Known Limitations

1. **Class Imbalance:** 'Mug' class heavily overrepresented (1058 vs others)
2. **Missing Classes:** 5 classes have no data yet
3. **Insufficient Data:** 2 classes need more images
4. **Domain Variety:** All images from online sources; need real office photos for robustness

## Dataset Versions

- **v1.0** (Oct 9, 2025): Initial Kaggle downloads, 1247 images across 5 classes
- Future versions will include additional sources and custom captures

## Usage Notes

This dataset is designed for image classification tasks. Images are organized in the following structure:

```
data/raw/
├── mug/
├── water_bottle/
├── mobile_phone/
├── keyboard/
├── computer_mouse/
├── stapler/
├── pen_pencil/
├── notebook/
├── office_chair/
└── office_bin/
```

After train/val/test split, data will be reorganized to:

```
data/processed/
├── train/
│   ├── mug/
│   ├── water_bottle/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

*This dataset card will be updated as data collection and processing progresses.*