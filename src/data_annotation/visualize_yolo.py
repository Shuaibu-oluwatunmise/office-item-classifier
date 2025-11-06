import cv2
from pathlib import Path

def plot_yolo_boxes(img_path, label_path, output_path, class_names):
    """Draw YOLO format bounding boxes on image"""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}")
        return
    
    h, w = img.shape[:2]
    
    # Read labels
    if not label_path.exists():
        print(f"No label file for {img_path}")
        return
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Color map for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]
    
    # Draw each box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h
        
        # Convert to corner coordinates
        x1 = int(x_center - box_w/2)
        y1 = int(y_center - box_h/2)
        x2 = int(x_center + box_w/2)
        y2 = int(y_center + box_h/2)
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = class_names[class_id]
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw filled rectangle for label background
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(str(output_path), img)

# ============================================
# CONFIGURATION
# ============================================

yolo_data_folder = Path("../../Data")
class_names = ['bottle', 'chair', 'keyboard', 'monitor', 'mouse', 'mug', 'notebook', 'pen', 'printer', 'stapler']

# Create output folders
output_folder = Path('../../visualizations')
for split in ['train', 'val', 'test']:
    (output_folder / split).mkdir(parents=True, exist_ok=True)

# ============================================
# VISUALIZE ALL SPLITS
# ============================================

print("=" * 60)
print("Visualizing YOLO_DATA annotations...")
print("=" * 60)

total_visualized = 0

for split in ['train', 'val', 'test']:
    img_dir = yolo_data_folder / split / 'images'
    lbl_dir = yolo_data_folder / split / 'labels'
    output_dir = output_folder / split
    
    if not img_dir.exists():
        print(f"\n⚠️  {split}/ folder not found, skipping...")
        continue
    
    # Get all images
    images = list(img_dir.glob('*.*'))
    
    print(f"\n📸 Processing {split}/ ({len(images)} images)...")
    
    for img_file in images:
        label_file = lbl_dir / f"{img_file.stem}.txt"
        output_file = output_dir / img_file.name
        
        plot_yolo_boxes(img_file, label_file, output_file, class_names)
    
    total_visualized += len(images)
    print(f"   ✓ Saved {len(images)} visualizations to visualizations/{split}/")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print(f"✅ Done! Visualized {total_visualized} images")
print("=" * 60)
print(f"\n📁 Check the 'visualizations/' folder:")
print(f"   - visualizations/train/")
print(f"   - visualizations/val/")
print(f"   - visualizations/test/")
print("\n💡 Each class has a different color:")
for i, name in enumerate(class_names):
    print(f"   {i}. {name}")
print("=" * 60)