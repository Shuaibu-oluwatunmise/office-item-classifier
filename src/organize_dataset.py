"""
Data Organization Script for Office Items Dataset
Organizes raw downloaded images into train/val/test splits
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Set random seed for reproducibility
random.seed(42)

# Define class names
CLASSES = [
    'mug',
    'water_bottle',
    'mobile_phone',
    'keyboard',
    'computer_mouse',
    'stapler',
    'pen',
    'notebook',
    'office_chair',
    'office_bin',
    'laptop'
]

# Define paths
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_directory_structure():
    """Create the processed data directory structure"""
    for split in ['train', 'val', 'test']:
        for class_name in CLASSES:
            dir_path = PROCESSED_DATA_DIR / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    print("Directory structure created successfully.")

def organize_images():
    """
    Organize images from raw directory into train/val/test splits
    Assumes raw data is organized as: data/raw/class_name/*.jpg
    """
    
    stats = {
        'total': 0,
        'train': 0,
        'val': 0,
        'test': 0,
        'per_class': {}
    }
    
    for class_name in CLASSES:
        class_dir = RAW_DATA_DIR / class_name
        
        if not class_dir.exists():
            print(f"Warning: Directory not found for class '{class_name}'")
            continue
        
        # Get all image files
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png'))
        
        if len(image_files) == 0:
            print(f"Warning: No images found for class '{class_name}'")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split sizes
        total = len(image_files)
        train_size = int(total * TRAIN_RATIO)
        val_size = int(total * VAL_RATIO)
        
        # Split data
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        # Copy files to respective directories
        for img_file in train_files:
            dest = PROCESSED_DATA_DIR / 'train' / class_name / img_file.name
            shutil.copy2(img_file, dest)
        
        for img_file in val_files:
            dest = PROCESSED_DATA_DIR / 'val' / class_name / img_file.name
            shutil.copy2(img_file, dest)
        
        for img_file in test_files:
            dest = PROCESSED_DATA_DIR / 'test' / class_name / img_file.name
            shutil.copy2(img_file, dest)
        
        # Update statistics
        stats['per_class'][class_name] = {
            'total': total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
        stats['total'] += total
        stats['train'] += len(train_files)
        stats['val'] += len(val_files)
        stats['test'] += len(test_files)
        
        print(f"Processed {class_name}: {len(train_files)} train, "
              f"{len(val_files)} val, {len(test_files)} test")
    
    return stats

def print_statistics(stats):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total images: {stats['total']}")
    print(f"Training: {stats['train']} ({stats['train']/stats['total']*100:.1f}%)")
    print(f"Validation: {stats['val']} ({stats['val']/stats['total']*100:.1f}%)")
    print(f"Testing: {stats['test']} ({stats['test']/stats['total']*100:.1f}%)")
    print("\nPer-class breakdown:")
    print("-"*60)
    
    for class_name, counts in stats['per_class'].items():
        print(f"{class_name:20s} | Train: {counts['train']:4d} | "
              f"Val: {counts['val']:4d} | Test: {counts['test']:4d} | "
              f"Total: {counts['total']:4d}")
    print("="*60)

def main():
    """Main execution function"""
    print("Starting dataset organization...")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Split ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    print()
    
    # Create directory structure
    create_directory_structure()
    
    # Organize images
    stats = organize_images()
    
    # Print statistics
    if stats['total'] > 0:
        print_statistics(stats)
        print("\nDataset organization complete!")
    else:
        print("\nNo images were processed. Check your raw data directory.")

if __name__ == "__main__":
    main()