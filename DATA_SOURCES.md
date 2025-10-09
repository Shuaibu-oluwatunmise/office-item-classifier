# Data Sources and Attribution

This document tracks all data sources used in this project for proper attribution and reproducibility.

## Download Date
October 9, 2025

## Datasets Downloaded from Kaggle

### 1. Mouse and Keyboard Dataset
- **Source:** Kaggle
- **Dataset ID:** `vshalsgh/mouse-and-keyboard-dataset`
- **URL:** https://www.kaggle.com/datasets/vshalsgh/mouse-and-keyboard-dataset
- **License:** Check dataset page for specific license
- **Classes Used:** Mouse, Keyboard
- **Download Method:** Kaggle API
- **Images Obtained:**
  - Computer Mouse: 31 images
  - Keyboard: 79 images
- **Notes:** YOLO format dataset with train/test/valid splits

### 2. Cup and Mug Dataset
- **Source:** Kaggle
- **Dataset ID:** `malikusman1221/cup-mug-dataset`
- **URL:** https://www.kaggle.com/datasets/malikusman1221/cup-mug-dataset
- **License:** Check dataset page for specific license
- **Classes Used:** Cup, Mug (both mapped to 'mug' class)
- **Download Method:** Kaggle API
- **Images Obtained:**
  - Mug: 1044 images (from both Cup and Mug folders)
- **Notes:** Pre-organized with train/val splits

### 3. Bottles and Cups Dataset
- **Source:** Kaggle
- **Dataset ID:** `dataclusterlabs/bottles-and-cups-dataset`
- **URL:** https://www.kaggle.com/datasets/dataclusterlabs/bottles-and-cups-dataset
- **License:** Check dataset page for specific license
- **Classes Used:** Bottle (mapped to water_bottle), Cup (mapped to mug)
- **Download Method:** Kaggle API
- **Images Obtained:**
  - Water Bottle: 76 images
  - Mug: 14 images (added to mug class)
- **Notes:** Object detection format with XML annotations (Pascal VOC format)

### 4. Electronics Dataset (Mouse, Keyboard, Phone)
- **Source:** Kaggle
- **Dataset ID:** `dataclusterlabs/electronics-mouse-keyboard-image-dataset`
- **URL:** https://www.kaggle.com/datasets/dataclusterlabs/electronics-mouse-keyboard-image-dataset
- **License:** Check dataset page for specific license
- **Classes Used:** Mouse, Keyboard, Phone/Mobile
- **Download Method:** Kaggle API
- **Images Obtained:**
  - Mobile Phone: 3 images
  - Computer Mouse: Included in count above
  - Keyboard: Included in count above
- **Notes:** Object detection format with XML annotations

## Current Dataset Statistics (After Initial Download)

| Class Name      | Images Count | Status       |
|-----------------|--------------|--------------|
| Mug             | 1058         | Sufficient   |
| Keyboard        | 79           | Sufficient   |
| Water Bottle    | 76           | Sufficient   |
| Computer Mouse  | 31           | Borderline   |
| Mobile Phone    | 3            | Insufficient |
| Stapler         | 0            | Missing      |
| Pen/Pencil      | 0            | Missing      |
| Notebook        | 0            | Missing      |
| Office Chair    | 0            | Missing      |
| Office Bin      | 0            | Missing      |

## Data Processing Pipeline

1. **Download:** Used Kaggle API to download datasets automatically
2. **Extraction:** Datasets extracted to `data/downloads/` directory
3. **Organization:** Custom Python script (`smart_organize.py`) used to:
   - Parse XML annotations where applicable
   - Map source classes to target classes
   - Copy images to appropriate class folders in `data/raw/`
   - Handle different dataset formats (YOLO, Pascal VOC, pre-organized folders)
4. **Cleanup:** Removed `data/downloads/` after successful organization

## Next Steps

1. Download/collect data for missing classes:
   - Stapler
   - Pen/Pencil  
   - Notebook
   - Office Chair
   - Office Bin
2. Augment insufficient classes:
   - Mobile Phone (need ~47 more images)
   - Computer Mouse (need ~19 more images for 50+)
3. Add custom photos from real office environment
4. Split organized data into train/validation/test sets

## Ethical Considerations

- All datasets are publicly available on Kaggle
- No personal or sensitive data in images
- Proper attribution provided for all sources
- Images used solely for educational/academic purposes (PDE3802 module assessment)

## Acknowledgments

We acknowledge and thank the dataset creators on Kaggle for making their data publicly available:
- Vishal Singh (vshalsgh) - Mouse and Keyboard Dataset
- malikusman1221 - Cup and Mug Dataset
- DataCluster Labs - Bottles/Cups and Electronics Datasets