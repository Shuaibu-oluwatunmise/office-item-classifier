"""
Smart Dataset Organizer
Handles different dataset formats and organizes them into raw class folders
"""

import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

RAW_DATA_DIR = Path('data/raw')
DOWNLOADS_DIR = Path('data/downloads')

def organize_cups_mugs():
    """
    Organize cups_mugs dataset (already has train/val/Cup/Mugs structure)
    """
    print("\nOrganizing cups_mugs dataset...")
    
    dataset_dir = DOWNLOADS_DIR / 'cups_mugs' / 'Cup-mug_data'
    stats = {'mug': 0}
    
    if not dataset_dir.exists():
        print("  cups_mugs dataset not found")
        return stats
    
    # Get images from both train and val, both Cup and Mugs folders
    for split in ['train', 'val']:
        for class_folder in ['Cup', 'Mugs']:
            source_dir = dataset_dir / split / class_folder
            if source_dir.exists():
                images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
                
                # Copy to mug class (treating both cup and mug as mug)
                target_dir = RAW_DATA_DIR / 'mug'
                target_dir.mkdir(exist_ok=True)
                
                for img in images:
                    dest = target_dir / f"cups_mugs_{split}_{class_folder}_{img.name}"
                    if not dest.exists():
                        shutil.copy2(img, dest)
                        stats['mug'] += 1
                
                print(f"  Copied {len(images)} images from {split}/{class_folder}")
    
    return stats

def organize_mouse_keyboard():
    """
    Organize mouse_keyboard dataset (YOLO format with train/test/valid)
    Note: This is object detection data, we'll just copy images
    """
    print("\nOrganizing mouse_keyboard dataset...")
    
    dataset_dir = DOWNLOADS_DIR / 'mouse_keyboard' / 'Mouse N Keyboard.v2i.yolov8'
    stats = {}
    
    if not dataset_dir.exists():
        print("  mouse_keyboard dataset not found")
        return stats
    
    # Copy all images - we'll manually sort them later or use the labels
    # For now, let's read the YAML to understand classes
    yaml_file = dataset_dir / 'data.yaml'
    
    # Just copy images from train/valid/test for manual inspection
    all_images = []
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_dir / split / 'images'
        if img_dir.exists():
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            all_images.extend(images)
    
    print(f"  Found {len(all_images)} images total")
    print(f"  Note: These need manual sorting into mouse/keyboard classes")
    print(f"  Images are in: {dataset_dir}")
    
    return stats

def parse_xml_annotation(xml_file):
    """
    Parse Pascal VOC XML annotation to get object classes
    Returns list of class names found in the image
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        classes = []
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is not None:
                classes.append(name.text.lower())
        
        return classes
    except:
        return []

def organize_bottles_cups():
    """
    Organize bottles_cups dataset (has XML annotations)
    """
    print("\nOrganizing bottles_cups dataset...")
    
    dataset_dir = DOWNLOADS_DIR / 'bottles_cups'
    img_dir = dataset_dir / 'images' / 'images'
    ann_dir = dataset_dir / 'annotation' / 'Bottles and Cups anotated'
    
    stats = {'water_bottle': 0, 'mug': 0}
    
    if not img_dir.exists() or not ann_dir.exists():
        print("  bottles_cups dataset structure not found")
        return stats
    
    # Process each image with its annotation
    xml_files = list(ann_dir.glob('*.xml'))
    
    for xml_file in xml_files:
        # Get corresponding image
        img_name = xml_file.stem + '.jpg'
        img_path = img_dir / img_name
        
        if not img_path.exists():
            continue
        
        # Parse XML to get classes
        classes = parse_xml_annotation(xml_file)
        
        # Copy to appropriate folders based on detected classes
        if 'bottle' in classes or 'water bottle' in classes:
            target_dir = RAW_DATA_DIR / 'water_bottle'
            target_dir.mkdir(exist_ok=True)
            dest = target_dir / f"bottles_{img_path.name}"
            if not dest.exists():
                shutil.copy2(img_path, dest)
                stats['water_bottle'] += 1
        
        if 'cup' in classes or 'mug' in classes:
            target_dir = RAW_DATA_DIR / 'mug'
            target_dir.mkdir(exist_ok=True)
            dest = target_dir / f"bottles_{img_path.name}"
            if not dest.exists():
                shutil.copy2(img_path, dest)
                stats['mug'] += 1
    
    print(f"  Processed {len(xml_files)} annotated images")
    
    return stats

def organize_electronics():
    """
    Organize electronics dataset (has XML annotations)
    """
    print("\nOrganizing electronics dataset...")
    
    dataset_dir = DOWNLOADS_DIR / 'electronics'
    img_dir = dataset_dir / 'samples_for_clients' / 'samples_for_clients'
    ann_dir = dataset_dir / 'annotations' / 'annotations'
    
    stats = {'computer_mouse': 0, 'keyboard': 0, 'mobile_phone': 0}
    
    if not img_dir.exists() or not ann_dir.exists():
        print("  electronics dataset structure not found")
        return stats
    
    # Process each image with its annotation
    xml_files = list(ann_dir.glob('*.xml'))
    
    for xml_file in xml_files:
        # Get corresponding image
        img_name = xml_file.stem  # Remove .jpg.xml
        img_path = img_dir / f"{img_name}.jpg"
        
        if not img_path.exists():
            continue
        
        # Parse XML to get classes
        classes = parse_xml_annotation(xml_file)
        
        # Map classes to our target classes
        class_mapping = {
            'mouse': 'computer_mouse',
            'keyboard': 'keyboard',
            'phone': 'mobile_phone',
            'mobile': 'mobile_phone',
            'mobile phone': 'mobile_phone'
        }
        
        # Copy to appropriate folders
        for detected_class in classes:
            target_class = class_mapping.get(detected_class.lower())
            
            if target_class:
                target_dir = RAW_DATA_DIR / target_class
                target_dir.mkdir(exist_ok=True)
                dest = target_dir / f"electronics_{img_path.name}"
                if not dest.exists():
                    shutil.copy2(img_path, dest)
                    stats[target_class] += 1
    
    print(f"  Processed {len(xml_files)} annotated images")
    
    return stats

def print_summary(all_stats):
    """Print summary of organized data"""
    print("\n" + "="*60)
    print("ORGANIZATION SUMMARY")
    print("="*60)
    
    total_images = 0
    for class_name in sorted(set(cls for stats in all_stats.values() for cls in stats.keys())):
        count = sum(stats.get(class_name, 0) for stats in all_stats.values())
        total_images += count
        print(f"  {class_name:20s}: {count:4d} images")
    
    print("-"*60)
    print(f"  Total organized: {total_images} images")
    print("="*60)
    
    # Show what we still need
    target_classes = ['mug', 'water_bottle', 'mobile_phone', 'keyboard', 
                     'computer_mouse', 'stapler', 'pen_pencil', 'notebook',
                     'office_chair', 'office_bin']
    
    print("\nClasses still needing data:")
    for cls in target_classes:
        count = sum(stats.get(cls, 0) for stats in all_stats.values())
        if count == 0:
            print(f"  {cls:20s}: NO DATA YET")
        elif count < 50:
            print(f"  {cls:20s}: {count:4d} images (need at least 50)")

def main():
    """Main execution"""
    print("="*60)
    print("SMART DATASET ORGANIZER")
    print("="*60)
    
    all_stats = {}
    
    # Organize each dataset
    all_stats['cups_mugs'] = organize_cups_mugs()
    all_stats['bottles_cups'] = organize_bottles_cups()
    all_stats['electronics'] = organize_electronics()
    all_stats['mouse_keyboard'] = organize_mouse_keyboard()
    
    # Print summary
    print_summary(all_stats)
    
    print("\nNext steps:")
    print("  1. Check data/raw/ folders for organized images")
    print("  2. Find datasets for missing classes (stapler, notebook, chair, bin)")
    print("  3. Add your own photos from class")
    print("  4. Run: python src/organize_dataset.py to split into train/val/test")

if __name__ == "__main__":
    main()