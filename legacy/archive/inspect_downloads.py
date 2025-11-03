"""
Inspect Downloaded Datasets
Explores the structure of downloaded Kaggle datasets
"""

import os
from pathlib import Path
import json

DOWNLOADS_DIR = Path('data/downloads')

def explore_directory(directory, max_depth=3, current_depth=0):
    """
    Recursively explore directory structure
    """
    if current_depth >= max_depth:
        return
    
    indent = "  " * current_depth
    
    try:
        items = sorted(directory.iterdir())
        
        for item in items[:20]:  # Limit to first 20 items per level
            if item.is_dir():
                print(f"{indent}📁 {item.name}/")
                explore_directory(item, max_depth, current_depth + 1)
            else:
                size_kb = item.stat().st_size / 1024
                print(f"{indent}📄 {item.name} ({size_kb:.1f} KB)")
        
        if len(items) > 20:
            print(f"{indent}... and {len(items) - 20} more items")
            
    except PermissionError:
        print(f"{indent}[Permission Denied]")

def count_files_by_extension(directory):
    """
    Count files by extension in directory
    """
    extensions = {}
    
    for item in directory.rglob('*'):
        if item.is_file():
            ext = item.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
    
    return extensions

def check_for_labels(directory):
    """
    Check for common label files (CSV, JSON, TXT, YAML)
    """
    label_files = []
    
    for pattern in ['*.csv', '*.json', '*.txt', '*.yaml', '*.yml', '*.xml']:
        label_files.extend(directory.rglob(pattern))
    
    return label_files

def main():
    """Explore all downloaded datasets"""
    
    if not DOWNLOADS_DIR.exists():
        print("No downloads directory found. Run download script first.")
        return
    
    datasets = [d for d in DOWNLOADS_DIR.iterdir() if d.is_dir()]
    
    if not datasets:
        print("No datasets found in downloads directory.")
        return
    
    print("="*70)
    print("DOWNLOADED DATASETS STRUCTURE")
    print("="*70)
    
    for dataset_dir in datasets:
        print(f"\n📦 Dataset: {dataset_dir.name}")
        print("-"*70)
        
        # Show structure
        print("\nDirectory Structure:")
        explore_directory(dataset_dir, max_depth=3)
        
        # Count files
        print("\nFile Types:")
        extensions = count_files_by_extension(dataset_dir)
        for ext, count in sorted(extensions.items()):
            print(f"  {ext if ext else '[no extension]':15s}: {count:4d} files")
        
        # Check for label files
        label_files = check_for_labels(dataset_dir)
        if label_files:
            print("\nFound potential label files:")
            for lf in label_files[:5]:  # Show first 5
                print(f"  📋 {lf.relative_to(dataset_dir)}")
            if len(label_files) > 5:
                print(f"  ... and {len(label_files) - 5} more")
        
        print("\n" + "="*70)
    
    print("\nNext steps:")
    print("  - Check if datasets have train/test/val folders")
    print("  - Look for CSV/JSON files that might contain labels")
    print("  - Manually inspect a few images to understand structure")

if __name__ == "__main__":
    main()