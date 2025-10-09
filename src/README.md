# Source Scripts Documentation

This directory contains scripts used for data collection, organization, and model development.

## Data Collection Scripts

### `download_datasets.py`
**Status:** Archive (not needed after initial download)  
**Purpose:** Helper script that provides download instructions and creates initial folder structure.  
**Usage:** `python src/download_datasets.py`  
**Note:** Used once during initial setup. Can be deleted or archived.

### `download_kaggle_datasets.py`
**Status:** Archive (not needed after initial download)  
**Purpose:** Automated Kaggle dataset downloader using Kaggle API.  
**Usage:** `python src/download_kaggle_datasets.py`  
**Note:** Successfully downloaded 4 datasets. Can be deleted or archived after data collection is complete.

### `inspect_downloads.py`
**Status:** Archive (used for debugging)  
**Purpose:** Explores downloaded dataset structure to understand folder organization.  
**Usage:** `python src/inspect_downloads.py`  
**Note:** Helped debug dataset formats. Can be deleted after organization is verified.

### `smart_organize.py`
**Status:** Archive (used once)  
**Purpose:** Intelligent organizer that handles multiple dataset formats (YOLO, Pascal VOC, pre-organized).  
**Usage:** `python src/smart_organize.py`  
**Note:** Successfully organized 1247 images. Can be deleted after organization is complete.

## Active Scripts

### `organize_dataset.py`
**Status:** ACTIVE - WILL BE USED  
**Purpose:** Splits raw data into train/validation/test sets (70/15/15 split).  
**Usage:** `python src/organize_dataset.py`  
**When to use:** After all data collection is complete, before training.  
**Output:** Creates `data/processed/train/`, `data/processed/val/`, `data/processed/test/`

## Future Scripts (To Be Created)

### `train_model.py`
**Purpose:** Training script for PyTorch image classification model  
**Status:** Not yet created

### `inference.py`
**Purpose:** Run inference on single images or camera feed  
**Status:** Not yet created

### `evaluate.py`
**Purpose:** Evaluate model on test set, generate confusion matrix and metrics  
**Status:** Not yet created

## Script Cleanup Recommendations

After data collection is complete and data is split, you can safely delete:
- `download_datasets.py`
- `download_kaggle_datasets.py`
- `inspect_downloads.py`
- `smart_organize.py`

Keep:
- `organize_dataset.py` (needed before training)

Or move old scripts to an `archive/` folder for documentation purposes.