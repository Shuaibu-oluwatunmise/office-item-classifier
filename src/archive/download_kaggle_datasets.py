"""
Automated Kaggle Dataset Downloader
Downloads and organizes datasets for office item classification
"""

import os
import zipfile
import shutil
from pathlib import Path
import subprocess

# Define datasets to download
DATASETS = {
    'mouse_keyboard': {
        'kaggle_id': 'vshalsgh/mouse-and-keyboard-dataset',
        'classes_mapping': {
            'Mouse': 'computer_mouse',
            'Keyboard': 'keyboard'
        }
    },
    'cups_mugs': {
        'kaggle_id': 'malikusman1221/cup-mug-dataset',
        'classes_mapping': {
            'Mug': 'mug',
            'Cup': 'mug'  # Map cup to mug class
        }
    },
    'bottles_cups': {
        'kaggle_id': 'dataclusterlabs/bottles-and-cups-dataset',
        'classes_mapping': {
            'Bottle': 'water_bottle',
            'Water Bottle': 'water_bottle'
        }
    },
    'electronics': {
        'kaggle_id': 'dataclusterlabs/electronics-mouse-keyboard-image-dataset',
        'classes_mapping': {
            'Mouse': 'computer_mouse',
            'Keyboard': 'keyboard',
            'Phone': 'mobile_phone',
            'Mobile': 'mobile_phone'
        }
    }
}

# Paths
RAW_DATA_DIR = Path('data/raw')
DOWNLOADS_DIR = Path('data/downloads')

def setup_directories():
    """Create necessary directories"""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Directories created successfully.")

def download_dataset(dataset_name, kaggle_id):
    """
    Download a dataset from Kaggle using kaggle CLI
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"Kaggle ID: {kaggle_id}")
    print(f"{'='*60}")
    
    dataset_dir = DOWNLOADS_DIR / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    try:
        # Use kaggle CLI to download
        cmd = f'kaggle datasets download -d {kaggle_id} -p {dataset_dir} --unzip'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded: {dataset_name}")
            return True
        else:
            print(f"Error downloading {dataset_name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Exception while downloading {dataset_name}: {e}")
        return False

def find_images_recursive(directory):
    """
    Recursively find all image files in a directory
    Returns list of Path objects
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    for ext in image_extensions:
        images.extend(Path(directory).rglob(f'*{ext}'))
        images.extend(Path(directory).rglob(f'*{ext.upper()}'))
    
    return images

def organize_dataset_images(dataset_name, classes_mapping):
    """
    Organize downloaded images into raw data class folders
    """
    print(f"\nOrganizing {dataset_name} images...")
    
    dataset_dir = DOWNLOADS_DIR / dataset_name
    
    if not dataset_dir.exists():
        print(f"Warning: {dataset_dir} does not exist. Skipping.")
        return
    
    stats = {}
    
    # Try to find class-based folder structure
    for source_class, target_class in classes_mapping.items():
        # Try different possible folder names (case-insensitive)
        possible_names = [source_class, source_class.lower(), source_class.upper(), 
                         source_class.replace(' ', '_'), source_class.replace(' ', '-')]
        
        found = False
        for name in possible_names:
            source_dir = dataset_dir / name
            if source_dir.exists() and source_dir.is_dir():
                found = True
                target_dir = RAW_DATA_DIR / target_class
                target_dir.mkdir(exist_ok=True)
                
                # Find all images in this class folder
                images = find_images_recursive(source_dir)
                
                # Copy images
                copied = 0
                for img_path in images:
                    # Create unique filename to avoid collisions
                    new_name = f"{dataset_name}_{img_path.stem}{img_path.suffix}"
                    dest_path = target_dir / new_name
                    
                    # Copy only if not exists
                    if not dest_path.exists():
                        shutil.copy2(img_path, dest_path)
                        copied += 1
                
                stats[target_class] = stats.get(target_class, 0) + copied
                print(f"  Copied {copied} images from '{source_class}' to '{target_class}'")
                break
        
        if not found:
            # If no class folder found, search for images in root
            images = find_images_recursive(dataset_dir)
            if images:
                print(f"  Warning: Could not find class folder for '{source_class}'")
                print(f"  Found {len(images)} images in dataset root. Manual sorting needed.")
    
    return stats

def print_summary(all_stats):
    """Print summary of downloaded and organized data"""
    print("\n" + "="*60)
    print("DOWNLOAD AND ORGANIZATION SUMMARY")
    print("="*60)
    
    total_images = 0
    for class_name in sorted(set(cls for stats in all_stats.values() for cls in stats.keys())):
        count = sum(stats.get(class_name, 0) for stats in all_stats.values())
        total_images += count
        print(f"  {class_name:20s}: {count:4d} images")
    
    print("-"*60)
    print(f"  Total downloaded: {total_images} images")
    print("="*60)
    
    # Check which classes still need data
    target_classes = ['mug', 'water_bottle', 'mobile_phone', 'keyboard', 
                     'computer_mouse', 'stapler', 'pen_pencil', 'notebook',
                     'office_chair', 'office_bin']
    
    print("\nClasses that still need more data:")
    for cls in target_classes:
        count = sum(stats.get(cls, 0) for stats in all_stats.values())
        if count < 50:
            print(f"  {cls:20s}: {count:4d} images (need at least 50)")

def main():
    """Main execution function"""
    print("="*60)
    print("AUTOMATED KAGGLE DATASET DOWNLOADER")
    print("="*60)
    print("\nThis script will download and organize office item datasets")
    print("from Kaggle. Make sure you have set up your Kaggle API credentials.")
    print("\nPress Ctrl+C to cancel, or wait 3 seconds to continue...")
    
    import time
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        return
    
    # Setup directories
    setup_directories()
    
    # Download each dataset
    all_stats = {}
    successful_downloads = 0
    
    for dataset_name, info in DATASETS.items():
        success = download_dataset(dataset_name, info['kaggle_id'])
        
        if success:
            successful_downloads += 1
            stats = organize_dataset_images(dataset_name, info['classes_mapping'])
            if stats:
                all_stats[dataset_name] = stats
    
    # Print summary
    print(f"\n\nSuccessfully downloaded {successful_downloads}/{len(DATASETS)} datasets")
    
    if all_stats:
        print_summary(all_stats)
        print("\n\nNext steps:")
        print("  1. Check data/raw/ folders for downloaded images")
        print("  2. Add more images for classes that need them")
        print("  3. Run: python src/organize_dataset.py to split into train/val/test")
    else:
        print("\nNo images were organized. Check for errors above.")

if __name__ == "__main__":
    main()