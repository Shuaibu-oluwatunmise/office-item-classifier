"""
Dataset Download Helper
Instructions and utilities for downloading office item datasets
"""

import os
from pathlib import Path

# Define class names
CLASSES = [
    'mug',
    'water_bottle',
    'mobile_phone',
    'keyboard',
    'computer_mouse',
    'stapler',
    'pen_pencil',
    'notebook',
    'office_chair',
    'office_bin'
]

def print_download_instructions():
    """
    Print instructions for manually downloading datasets
    """
    
    print("="*70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print()
    print("We need images for the following 10 classes:")
    for i, class_name in enumerate(CLASSES, 1):
        print(f"  {i}. {class_name.replace('_', ' ').title()}")
    print()
    print("-"*70)
    print("RECOMMENDED SOURCES:")
    print("-"*70)
    print()
    
    print("1. ROBOFLOW UNIVERSE")
    print("   URL: https://universe.roboflow.com/")
    print("   Search for: 'office objects', 'desk items', 'office equipment'")
    print("   Format: Download as 'Folder Structure' or 'COCO'")
    print()
    
    print("2. KAGGLE DATASETS")
    print("   URL: https://www.kaggle.com/datasets")
    print("   Search for individual classes or office items collections")
    print("   You may need: kaggle API key for automated download")
    print()
    
    print("3. GOOGLE OPEN IMAGES")
    print("   URL: https://storage.googleapis.com/openimages/web/index.html")
    print("   Search by class name")
    print("   Use OIDv4_ToolKit for batch download")
    print()
    
    print("4. IMAGENET (if available)")
    print("   Many office items have ImageNet categories")
    print("   Requires ImageNet access")
    print()
    
    print("-"*70)
    print("DOWNLOAD GUIDELINES:")
    print("-"*70)
    print("  - Target: 50-150 images per class minimum")
    print("  - Ensure variety: different angles, lighting, backgrounds")
    print("  - Check image quality: remove blurry or corrupted images")
    print("  - Verify labels: ensure images match their class")
    print()
    print("-"*70)
    print("FOLDER STRUCTURE FOR RAW DATA:")
    print("-"*70)
    print("  data/raw/")
    for class_name in CLASSES:
        print(f"    ├── {class_name}/")
        print(f"    │   ├── image_001.jpg")
        print(f"    │   ├── image_002.jpg")
        print(f"    │   └── ...")
    print()
    print("After downloading, place all images in their respective class folders")
    print("under data/raw/, then run: python src/organize_dataset.py")
    print("="*70)

def check_raw_data_status():
    """
    Check current status of raw data directory
    """
    raw_dir = Path('data/raw')
    
    print("\n" + "="*70)
    print("CURRENT RAW DATA STATUS")
    print("="*70)
    
    if not raw_dir.exists():
        print("Raw data directory does not exist yet.")
        print("Create it with: mkdir data/raw")
        return
    
    total_images = 0
    
    for class_name in CLASSES:
        class_dir = raw_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg')) + \
                    list(class_dir.glob('*.jpeg')) + \
                    list(class_dir.glob('*.png'))
            count = len(images)
            total_images += count
            status = "OK" if count >= 50 else "NEEDS MORE"
            print(f"  {class_name:20s}: {count:4d} images  [{status}]")
        else:
            print(f"  {class_name:20s}:    0 images  [MISSING]")
    
    print("-"*70)
    print(f"Total images: {total_images}")
    
    if total_images == 0:
        print("\nNo images found. Please download datasets following the instructions above.")
    elif total_images < 500:
        print("\nYou have some images, but consider adding more for better model performance.")
    else:
        print("\nGood amount of data collected! Ready to organize with organize_dataset.py")
    
    print("="*70)

def create_class_directories():
    """
    Create empty class directories in raw data folder
    """
    raw_dir = Path('data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in CLASSES:
        class_dir = raw_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    print("Created class directories in data/raw/")
    print("You can now place downloaded images in their respective folders.")

def main():
    """Main execution"""
    print_download_instructions()
    print()
    
    # Create directories if they don't exist
    create_class_directories()
    
    # Check current status
    check_raw_data_status()
    
    print("\nNext steps:")
    print("  1. Download images following the instructions above")
    print("  2. Place images in data/raw/<class_name>/ folders")
    print("  3. Run: python src/organize_dataset.py to split into train/val/test")

if __name__ == "__main__":
    main()