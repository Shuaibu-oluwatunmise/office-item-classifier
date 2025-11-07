# Source Code Documentation

This document provides comprehensive documentation for all scripts and modules in the `src/` directory of the Office Item Classifier project.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Architecture](#project-architecture)
3. [Data Collection](#data-collection)
4. [Data Annotation](#data-annotation)
5. [Detection Pipeline](#detection-pipeline)
6. [Classification Pipeline](#classification-pipeline)
7. [User Interface](#user-interface)
8. [Complete Workflows](#complete-workflows)

---

## Overview

The `src/` folder contains five main modules that form a complete end-to-end pipeline:

```
src/
├── data_collection/      # Video recording and frame extraction
├── data_annotation/      # Automatic labeling with Grounding DINO
├── detection/            # Object detection (YOLOv8)
├── classification/       # Image classification (YOLOv8-cls)
└── ui/                   # Tkinter GUI application
```

**Complete Pipeline Flow:**
```
Record Videos → Extract Frames → Auto-Annotate → Train Models → Deploy via UI
```

---

## Project Architecture

### Technology Stack

**Deep Learning:**
- YOLOv8 (Ultralytics) - Detection and classification
- Grounding DINO - Zero-shot object detection for annotation
- PyTorch - Deep learning framework

**Computer Vision:**
- OpenCV (cv2) - Image/video processing
- PIL (Pillow) - Image manipulation
- NumPy - Array operations

**User Interface:**
- Tkinter - GUI framework
- ttk - Themed widgets

**Utilities:**
- tqdm - Progress bars
- pathlib - File path handling

### Hardware Requirements

**Training & Annotation:**
- GPU: NVIDIA RTX 5080 (or any CUDA-compatible GPU)
- CPU: Intel Core Ultra 7 265K (20 cores) or equivalent
- RAM: 64GB recommended (16GB minimum)

**Inference:**
- GPU: Optional (CUDA 12.8+)
- CPU: Any modern processor
- RAM: 8GB minimum

### Data Formats

**Detection Dataset (YOLO format):**
```
Data/
├── train/
│   ├── images/          # Training images
│   └── labels/          # YOLO format (.txt)
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml            # Dataset configuration
```

**YOLO Label Format (.txt):**
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
```
- All values normalized (0-1)
- One line per object
- Multiple objects = multiple lines

**Classification Dataset (folder-based):**
```
Classification_Data/
├── train/
│   ├── bottle/          # All bottle training images
│   ├── chair/
│   └── ...
├── val/
│   ├── bottle/
│   └── ...
└── test/
    ├── bottle/
    └── ...
```
- Folder name = class label
- No label files needed
- One image = one class

---

## Data Collection

**Location:** `src/data_collection/`

### Purpose
Collect training data by recording short videos of objects and extracting frames for training.

### Scripts

#### 1. `data_collection.py` - Video Recording Tool

**Purpose:** Interactive CLI tool for recording videos of objects using webcam.

**Features:**
- User-friendly prompt loop
- 20-second recording duration
- Automatic folder organization
- Live preview with countdown timer
- Timestamp-based naming

**Usage:**
```bash
cd src/data_collection
python data_collection.py
```

**Interactive Flow:**
```
Enter the class name: bottle
[Camera opens with countdown timer]
[Records for 20 seconds]
Video saved to: Data_Video/bottle/video_20251106_133245.avi

Enter the class name: chair
[Records another video]
```

**Configuration:**
- **Camera index:** 1 (external webcam)
- **Duration:** 20 seconds (hardcoded)
- **Codec:** XVID (.avi format)
- **FPS:** Auto-detected from camera
- **Output:** `Data_Video/<class_name>/video_YYYYMMDD_HHMMSS.avi`

**Naming Convention:**
- All class names converted to lowercase
- Timestamp prevents filename conflicts
- Multiple videos per class allowed

**Controls:**
- Press 'q' during recording to stop early
- Type 'quit' or 'exit' to close application

**Design Rationale:**
- **20 seconds:** Sufficient for ~600 frames at 30 FPS
- **Video-first approach:** Easier to capture movement and different perspectives
- **External webcam:** Better quality than built-in laptop cameras

---

#### 2. `extract_frames.py` - Frame Extraction Pipeline

**Purpose:** Convert recorded videos into individual image frames for training.

**Features:**
- GPU-accelerated OpenCV processing
- Batch processing of all classes
- Every frame extraction (30 FPS)
- High-quality JPEG output
- Progress tracking with tqdm
- Comprehensive statistics

**Usage:**
```bash
python extract_frames.py
```

**Automatic Processing:**
```
Input:  Data_Video/bottle/video_20251106_133245.avi (20s, 590 frames)
Output: images/bottle/bottle_0001.jpg ... bottle_0590.jpg

Input:  Data_Video/bottle/video_20251106_133342.avi (20s, 589 frames)
Output: images/bottle/bottle_0591.jpg ... bottle_1179.jpg
```

**Configuration:**
- **Frame skip:** 1 (extracts every frame)
- **Output quality:** 95% JPEG compression
- **Naming:** `<classname>_<number>.jpg`
- **GPU acceleration:** Enabled if available

**Output Structure:**
```
images/
├── bottle/
│   ├── bottle_0001.jpg
│   ├── bottle_0002.jpg
│   └── ... (1,179 total)
├── chair/
│   └── ...
└── ...
```

**Statistics Displayed:**
- FPS and total frames per video
- Duration of each video
- Frames extracted per video
- Total images per class
- Overall extraction summary

**Design Rationale:**
- **Every frame extraction:** Maximizes data from each video
- **Continuous numbering:** Treats multiple videos as one dataset
- **GPU acceleration:** Speeds up video decoding
- **High JPEG quality:** Preserves visual details for training

**Performance:**
- ~30 frames/second extraction speed
- 20-second video → ~20 seconds to process
- 2 videos per class → ~40 seconds per class
- 10 classes → ~7 minutes total

---

### Data Collection Workflow

**Step-by-step process:**

1. **Record Videos:**
   ```bash
   python data_collection.py
   # Record 2 videos per class (bottle, chair, keyboard, etc.)
   ```

2. **Extract Frames:**
   ```bash
   python extract_frames.py
   # Converts all videos to images automatically
   ```

3. **Result:**
   - ~1,200 images per class
   - Ready for annotation
   - Organized by class folders

**Expected Output:**
- 10 classes × 1,200 images = **12,000 raw training images**
- Time investment: ~20 minutes total (10 classes × 2 videos × 1 minute each)

---

## Data Annotation

**Location:** `src/data_annotation/`

### Purpose
Automatically generate bounding box labels for object detection using Grounding DINO, eliminating manual annotation work.

### Standalone Repository
This module is also available as a standalone tool:
- **GitHub:** https://github.com/Shuaibu-oluwatunmise/Auto-Annotation
- **Documentation:** Includes troubleshooting, fixes, and usage examples
- **Reusable:** Can be used for any object detection project

### Scripts

#### 1. `Auto_Annotate.py` - Zero-Shot Auto-Labeling Pipeline

**Purpose:** Complete automation from raw images to YOLO-ready dataset with bounding box labels.

**Features:**
- Zero-shot object detection (no training needed)
- Text-based prompts (describe objects in plain English)
- Automatic class ID remapping
- Stratified dataset splitting (80/10/10)
- YOLO format output
- GPU-accelerated processing

**Usage:**
```bash
cd src/data_annotation
python Auto_Annotate.py
```

**5-Step Pipeline:**

**Step 1: File Standardization**
- Renames all images to `classname0001.jpg`, `classname0002.jpg`, etc.
- Ensures consistent naming for downstream processing

**Step 2: Class Mapping**
- Maps folder names to text prompts for Grounding DINO
```python
prompt_mapping = {
    "bottle": "water bottle",       # More specific = better detection
    "mouse": "computer mouse",       # Disambiguates from animal
    "chair": "office chair",         # Context-specific
    "keyboard": "computer keyboard",
    "monitor": "computer monitor",
    "mug": "coffee mug",
    "notebook": "notebook",
    "pen": "pen",
    "printer": "printer",
    "stapler": "stapler"
}
```
- Assigns sequential class IDs (0, 1, 2, ...)

**Step 3: AI Annotation**
- Grounding DINO detects objects using text prompts
- Generates bounding boxes automatically
- Creates YOLO format labels
- Remaps class IDs to maintain consistency across all classes

**Step 4: Dataset Merging & Splitting**
- Combines all annotated classes
- Stratified 80/10/10 split (train/val/test)
- Maintains class balance across splits
- Random seed 42 for reproducibility

**Step 5: YOLO Configuration**
- Creates `data.yaml` with paths and class names
```yaml
path: /absolute/path/to/Data
train: train/images
val: val/images
test: test/images
nc: 10
names: ['bottle', 'chair', 'keyboard', 'monitor', 'mouse', 'mug', 'notebook', 'pen', 'printer', 'stapler']
```

**Configuration:**
```python
# Input/Output
input_folder = Path("../data_collection/images")
output_folder = Path("../../Data")

# Split ratios
train_split = 0.8
val_split = 0.1
test_split = 0.1
```

**GPU Support:**
- Default: Uses available GPU (CUDA)
- Force CPU: Uncomment `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

**Performance:**
- **RTX 5080:** ~1,300 images/minute
- **26,000 images:** 20 minutes total
- **Manual equivalent:** 250+ hours

**Output Structure:**
```
Data/
├── train/
│   ├── images/      # 20,800 images (80%)
│   └── labels/      # 20,800 .txt files
├── val/
│   ├── images/      # 2,600 images (10%)
│   └── labels/      # 2,600 .txt files
├── test/
│   ├── images/      # 2,600 images (10%)
│   └── labels/      # 2,600 .txt files
└── data.yaml
```

**Label Quality:**
- YOLO format: `class_id x_center y_center width height`
- Normalized coordinates (0-1)
- One line per detected object
- Consistent class IDs across entire dataset

**Design Rationale:**
- **Grounding DINO:** Foundation model with zero-shot capabilities
- **Text prompts:** More intuitive than visual examples
- **Stratified splitting:** Ensures balanced class distribution
- **GPU acceleration:** 750× faster than manual annotation

---

#### 2. `visualize_yolo.py` - Annotation Quality Check

**Purpose:** Visual verification of auto-generated annotations before training.

**Features:**
- Draws bounding boxes on images
- Color-coded per class (10 distinct colors)
- Labels with class names
- Processes all splits (train/val/test)
- Saves to `visualizations/` folder

**Usage:**
```bash
python visualize_yolo.py
```

**Output:**
```
visualizations/
├── train/         # Annotated training images
├── val/           # Annotated validation images
└── test/          # Annotated test images
```

**Color Palette:**
- Class 0: Green
- Class 1: Blue
- Class 2: Red
- Class 3: Cyan
- Class 4: Magenta
- Class 5: Yellow
- Class 6: Purple
- Class 7: Orange
- Class 8: Teal
- Class 9: Olive

**Design Rationale:**
- **Visual inspection:** Quickly spot annotation issues
- **Quality assurance:** Verify before spending hours training
- **Debugging:** Identify problematic classes or images

---

### Data Annotation Workflow

**Complete process:**

1. **Prepare Images:**
   ```bash
   # Ensure images are in data_collection/images/<class_name>/
   ```

2. **Run Auto-Annotation:**
   ```bash
   cd src/data_annotation
   python Auto_Annotate.py
   ```
   **Time: 20 minutes for 26,000 images on RTX 5080**

3. **Visualize & Verify:**
   ```bash
   python visualize_yolo.py
   # Check visualizations/ folder for quality
   ```

4. **Result:**
   - Complete YOLO dataset ready for training
   - `Data/` folder with train/val/test splits
   - `data.yaml` configuration file

**Performance Impact:**
- Traditional manual annotation: **250+ hours**
- Auto-annotation (RTX 5080): **20 minutes**
- **Speedup: 750×**

**Cost Savings:**
- Professional annotators: $2,500-$5,000
- This system: $0 (+ ~$0.50 electricity)
- **ROI: Infinite**

---

### Troubleshooting

**Common Issues:**

**OpenCV Path Error (Python 3.12+):**
- See GitHub repo: https://github.com/Shuaibu-oluwatunmise/Auto-Annotation
- Fix available in repository

**CUDA Errors (RTX 40XX/50XX):**
- Force CPU mode: Uncomment line 2 in `Auto_Annotate.py`
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**Empty Annotations:**
- Check text prompts in `prompt_mapping`
- Try more descriptive prompts (e.g., "black computer mouse" instead of "mouse")
- Verify image quality (not too blurry or small)

**Slow Processing:**
- CPU mode is slow (~2-3 seconds per image)
- Consider Google Colab with free GPU
- Or use batch processing in smaller sets

---

## Detection Pipeline

**Location:** `src/detection/`

### Purpose
Train and deploy YOLOv8 object detection models that localize objects with bounding boxes.

### Required Data Format
Must have YOLO-formatted dataset from annotation step (see Data Annotation section).

### Scripts

#### 1. `train.py` - Detection Model Training

**Purpose:** Train YOLOv8n detection model on annotated dataset.

**Usage:**
```bash
cd src/detection
python train.py
```

**Configuration:**
```python
DATA_YAML = Path("Data/data.yaml")    # Dataset configuration
MODEL_NAME = "yolov8n.pt"             # Nano model (fastest)
EPOCHS = 50                           # Training epochs
IMG_SIZE = 640                        # Image size
BATCH_SIZE = 32                       # Batch size
PATIENCE = 10                         # Early stopping
```

**Training Details:**
- **Model:** YOLOv8n (nano) - smallest and fastest
- **Pretrained:** Starts from COCO-trained weights
- **Device:** Auto-detects CUDA/CPU
- **Workers:** 12 threads for data loading
- **AMP:** Automatic mixed precision (faster training)

**Output:**
```
runs/detect/yolov8n_detect_V5/
├── weights/
│   ├── best.pt          # Best validation model
│   └── last.pt          # Final epoch model
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── val_batch0_*.jpg     # Validation predictions
└── args.yaml            # Training arguments
```

**Training Metrics:**
- mAP50: Mean Average Precision at IoU 0.5
- mAP50-95: Mean Average Precision at IoU 0.5-0.95
- Precision: % of correct positive predictions
- Recall: % of actual positives detected
- Per-class metrics

**Expected Performance:**
- mAP50: ~98%
- mAP50-95: ~96%
- Training time: ~2 hours on RTX 5080
- Training time: ~8-10 hours on CPU (20 cores)

**Early Stopping:**
- Monitors validation mAP50
- Stops if no improvement for 10 epochs
- Prevents overfitting

---

#### 2. `check_labels.py` - Label Validation

**Purpose:** Find corrupted or malformed label files before training.

**Usage:**
```bash
python check_labels.py
```

**Validation:**
- Checks all .txt files in train/val/test labels
- Each line must have exactly 5 values
- Flags files with >5 values (corrupted)
- Reports all invalid files

**Output:**
```
⚠️ Invalid label files found:
 - Data/train/labels/Keyboard1172.txt
 - Data/train/labels/Pen1110.txt
```

**Why Needed:**
- Auto-annotation can occasionally produce bad labels
- Prevents training crashes
- Quality assurance before training

---

#### 3. `clean_invalid_labels.py` - Automated Cleanup

**Purpose:** Remove or fix bad annotations automatically.

**Usage:**
```bash
python clean_invalid_labels.py
```

**Actions:**
1. **Empty labels** → Delete label + corresponding image
2. **Partially bad** → Keep only valid lines (5 values each)
3. **Logging** → Records all actions to `cleanup_log.txt`

**Example Output:**
```
🧹 Cleanup started: 2025-11-05 20:34:00
============================================================
[DEL] Removed label: Data/train/labels/Keyboard1172.txt
     └─ Removed image: Data/train/images/Keyboard1172.jpg
[FIX] Cleaned bad lines in: Data/train/labels/Keyboard1261.txt
[DEL] Removed label: Data/train/labels/Pen1110.txt
     └─ Removed image: Data/train/images/Pen1110.jpg

📊 SUMMARY
   • Labels deleted: 45
   • Images deleted: 45
   • Labels cleaned: 8

✅ Cleanup complete!
```

**Cleanup Strategy:**
- Removes images without valid labels (can't train on them)
- Preserves images with at least one valid detection
- Maintains train/val/test integrity
- Logs everything for review

---

#### 4. `dataset_stats.py` - Dataset Analysis

**Purpose:** Generate comprehensive dataset statistics.

**Usage:**
```bash
python dataset_stats.py
```

**Statistics Displayed:**
```
📊 DATASET STATISTICS
============================================================

📂 TRAIN
   • Images: 20,755
   • Labels: 20,755
   • Object instances: 20,755

📂 VAL
   • Images: 2,595
   • Labels: 2,595
   • Object instances: 2,595

📂 TEST
   • Images: 2,596
   • Labels: 2,596
   • Object instances: 2,596

============================================================
📦 TOTAL IMAGES: 25,946
📄 TOTAL LABEL FILES: 25,946
🧩 TOTAL OBJECT INSTANCES: 25,946

🔢 CLASS DISTRIBUTION:
   • Class 0: 2,595 instances
   • Class 1: 2,594 instances
   • Class 2: 2,596 instances
   ...

✅ Dataset statistics complete!
```

**Use Cases:**
- Verify dataset balance
- Check for class imbalance
- Confirm cleanup results
- Document dataset composition

---

#### 5. `File_Inference.py` - Batch Detection

**Purpose:** Run detection on images/videos/folders.

**Usage:**
```bash
# Single image
python File_Inference.py -path/to/image.jpg

# Folder of images/videos
python File_Inference.py -path/to/folder

# Single video
python File_Inference.py -path/to/video.mp4
```

**Features:**
- **Folder mode:** Process all images/videos in folder
- **Single file:** Process one image or video
- **Confidence threshold:** 0.25 (adjustable)
- **Large text overlays:** Font scale 1.8, thickness 4
- **Color-coded classes:** Different color per class
- **Progress bars:** Visual feedback with tqdm

**Supported Formats:**
- Images: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- Videos: .mp4, .avi, .mov, .mkv, .wmv

**Output:**
- Saves to `detection_results/` folder
- Annotated images with bounding boxes
- Annotated videos with frame-by-frame overlays
- Preserves original filenames

**Visualization:**
- Bounding boxes: Class-specific colors
- Labels: Class name + confidence percentage
- Background rectangles for text readability
- Large, bold fonts for clarity

**Configuration:**
```python
model_path = 'runs/detect/yolov8n_detect_V5/weights/best.pt'
confidence_threshold = 0.25
iou_threshold = 0.45
font_scale = 1.8
box_thickness = 3
```

---

#### 6. `Live_Feed.py` - Real-time Detection

**Purpose:** Live webcam object detection with recording capability.

**Usage:**
```bash
python Live_Feed.py
```

**Architecture:**
- **Multi-threaded:** Separate detection thread
- **Detection interval:** 100ms (10 FPS detection)
- **Video display:** 30 FPS (smooth preview)
- **Auto-recording:** Starts immediately

**Features:**
- FPS counter and processing time display
- Object count overlay
- Recording status indicator
- Color-coded bounding boxes per class
- Threading prevents UI lag
- Save recordings with annotations

**Controls:**
- **SPACE:** Toggle recording on/off
- **R:** Restart recording (new file)
- **Q:** Quit application

**Output:**
- Saves to `Webcam_Detections/session_YYYYMMDD_HHMMSS.mp4`
- Includes all annotations in recorded video
- MP4 format with annotations baked in

**Performance Optimization:**
- Frame queue (max 2 frames)
- Non-blocking detection
- Drops frames if detection can't keep up
- Efficient threading model

**Display:**
```
┌─────────────────────────────────────────┐
│ FPS: 28.5 | Process: 95ms | Objects: 3 │
│ Rec: ON                                 │
│                                         │
│  [Live video feed with bounding boxes] │
│                                         │
│ SPACE: Toggle Rec | R: Restart | Q: Quit
└─────────────────────────────────────────┘
```

---

### Detection Workflow

**Complete process:**

1. **Prepare Dataset:**
   ```bash
   # Ensure Data/ folder exists with YOLO format
   # From data_annotation step
   ```

2. **Validate Labels:**
   ```bash
   cd src/detection
   python check_labels.py
   ```

3. **Clean Dataset:**
   ```bash
   python clean_invalid_labels.py
   # Review cleanup_log.txt
   ```

4. **Verify Statistics:**
   ```bash
   python dataset_stats.py
   # Check class balance
   ```

5. **Train Model:**
   ```bash
   python train.py
   # Wait ~2 hours (GPU) or ~8-10 hours (CPU)
   ```

6. **Run Inference:**
   ```bash
   # Batch processing
   python File_Inference.py -test_images/

   # Live camera
   python Live_Feed.py
   ```

---

## Classification Pipeline

**Location:** `src/classification/`

### Purpose
Train and deploy YOLOv8 image classification models that identify entire images into single classes.

### Required Data Format
Folder-based organization where folder name = class label (see below).

### Scripts

#### 1. `organise_dataset.py` - Dataset Preparation

**Purpose:** Convert detection dataset to classification format and balance classes.

**Usage:**
```bash
cd src/classification
python organise_dataset.py
```

**Process:**
1. **Source:** `Data/` (detection format with labels)
2. **Reads:** Images from class folders
3. **Caps:** Max 1,950 images per class (prevents imbalance)
4. **Excess handling:** Moves extras to `excesses/` folder
5. **Splits:** 80/10/10 train/val/test
6. **Output:** `Classification_Data/` folder structure

**Configuration:**
```python
SOURCE_DIR = Path("Data")                      # Detection dataset
DEST_DIR = Path("Classification_Data")          # Classification output
EXCESS_DIR = Path("excesses")                  # Overflow storage
MAX_IMAGES_PER_CLASS = 1950                    # Balance cap
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
```

**Output Structure:**
```
Classification_Data/
├── train/
│   ├── bottle/         # 1,560 images (80% of 1,950)
│   ├── chair/          # 1,560 images
│   └── ...
├── val/
│   ├── bottle/         # 195 images (10%)
│   └── ...
└── test/
    ├── bottle/         # 195 images (10%)
    └── ...

excesses/
├── bottle/             # Extras beyond 1,950
└── ...
```

**Why Needed:**
- Classification needs folder-based organization (no labels)
- Ensures class balance for fair training
- Preserves extra data for future use
- Prevents overfitting to overrepresented classes

**Example Output:**
```
📂 Class 'bottle': 2600 images
   🗑️ Moved 650 excess images to 'excesses/bottle'
   ✅ train: 1560 images
   ✅ val: 195 images
   ✅ test: 195 images

📂 Class 'chair': 2600 images
   🗑️ Moved 650 excess images to 'excesses/chair'
   ✅ train: 1560 images
   ✅ val: 195 images
   ✅ test: 195 images

✅ Dataset organization complete!
```

---

#### 2. `rename_dataset.py` - File Standardization

**Purpose:** Rename all images with consistent pattern for professional dataset.

**Usage:**
```bash
python rename_dataset.py
```

**Naming Convention:**
```
Before: IMG_3453.jpg, photo_1.png, DSC_0042.jpg
After:  bottle_0001_train.jpg, bottle_0002_train.jpg, ...
        bottle_0001_val.jpg, bottle_0001_test.jpg
```

**Benefits:**
- Clear split identification (_train, _val, _test suffix)
- Sequential numbering (0001, 0002, ...)
- Prevents name conflicts across splits
- Professional dataset organization
- Easier debugging and tracking

**Example Output:**
```
📂 Renaming 1560 images in 'train/bottle'...
📂 Renaming 195 images in 'val/bottle'...
📂 Renaming 195 images in 'test/bottle'...
...

✅ All images renamed successfully with split suffix!
```

---

#### 3. `train.py` - Classification Model Training

**Purpose:** Train YOLOv8n-cls classification model on organized dataset.

**Usage:**
```bash
python train.py
```

**Configuration:**
```python
DATA_DIR = Path("Classification_Data")
MODEL_NAME = "yolov8n-cls.pt"        # Classification variant
EPOCHS = 25                          # Fewer than detection
IMG_SIZE = 224                       # Standard for classification
BATCH_SIZE = 32
PATIENCE = 10                        # Early stopping
```

**Training Details:**
- **Model:** YOLOv8n-cls (classification-specific)
- **Image size:** 224×224 (smaller than detection's 640)
- **Epochs:** 25 (classification converges faster)
- **Device:** Auto-detects CUDA/CPU
- **Workers:** 12 threads

**Output:**
```
runs/classify/yolov8n_cls_V4/
├── weights/
│   ├── best.pt              # Best validation model
│   └── last.pt              # Final epoch model
├── results.png              # Training curves
├── confusion_matrix.png     # Confusion matrix
├── confusion_matrix_normalized.png
└── args.yaml                # Training arguments
```

**Training Metrics:**
- Top-1 Accuracy: % of correct predictions
- Top-5 Accuracy: % where true class in top 5
- Per-class accuracy
- Confusion matrix

**Expected Performance:**
- Top-1 Accuracy: ~99.9%
- Top-5 Accuracy: 100%
- Training time: ~1 hour on RTX 5080
- Training time: ~4-5 hours on CPU (20 cores)

**Early Stopping:**
- Monitors validation accuracy
- Stops if no improvement for 10 epochs
- Typical stopping: Epoch 15-20

**Key Differences from Detection:**
- Smaller image size (224 vs 640)
- Fewer epochs needed (25 vs 50)
- No bounding box prediction
- Simpler task = faster training

---

#### 4. `file_inference.py` - Batch Classification

**Purpose:** Classify images/videos/folders with trained model.

**Usage:**
```bash
# Single image
python file_inference.py -path/to/image.jpg

# Folder of images/videos
python file_inference.py -path/to/folder

# Single video
python file_inference.py -path/to/video.mp4
```

**Features:**
- **Folder mode:** Process all images/videos
- **Single file:** One image or video
- **Confidence threshold:** 0.3
- **PIL + OpenCV:** High-quality text rendering
- **Progress bars:** tqdm for user feedback

**Supported Formats:**
- Images: .jpg, .jpeg, .png, .bmp, .gif, .webp
- Videos: .mp4, .avi, .mov, .mkv, .wmv

**Output:**
- `results/` folder (created inside input folder)
- Images with class label + confidence overlay
- Videos with frame-by-frame classification
- Preserves original filenames with suffix

**Text Rendering:**
- Large font (60pt)
- Semi-transparent background
- High contrast for readability
- PIL for better text quality than OpenCV

**Configuration:**
```python
MODEL_PATH = "runs/classify/yolov8n_cls_V4/weights/best.pt"
FONT_SIZE = 60
CONF_THRESHOLD = 0.3
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 0, 180)
```

**Example Output:**
```
🖼️ Processing 50 image(s)...
Images: 100%|████████████████| 50/50 [00:15<00:00, 3.2img/s]

🎞️ Processing 2 video(s)...
video1.mp4: 100%|██████████| 450/450 [01:30<00:00, 5.0frame/s]

✅ All results saved to: /path/to/folder/results/
```

---

#### 5. `live_inference.py` - Real-time Classification

**Purpose:** Live webcam classification with instant predictions.

**Usage:**
```bash
python live_inference.py
```

**Features:**
- Camera index 1 (external webcam)
- Confidence threshold: 0.3
- PIL text rendering for quality
- Real-time class + confidence display
- Simple, clean interface

**Controls:**
- **Q:** Quit

**Display:**
```
┌─────────────────────────────┐
│ Mug - 98.45%                │
│                             │
│  [Live video feed]          │
│                             │
└─────────────────────────────┘
```

**Performance:**
- ~30 FPS video display
- Real-time inference (~50ms per frame)
- Low latency overlay
- Smooth predictions

**Simpler than Detection:**
- No bounding boxes
- Just one prediction per frame
- Lower computational cost
- Easier to read at a glance

**Configuration:**
```python
MODEL_PATH = "runs/classify/yolov8n_cls_V4/weights/best.pt"
FONT_SIZE = 36
CONF_THRESHOLD = 0.3
```

---

### Classification Workflow

**Complete process:**

1. **Prepare Dataset:**
   ```bash
   cd src/classification
   # Ensure Data/ exists from detection pipeline
   ```

2. **Organize & Balance:**
   ```bash
   python organise_dataset.py
   # Creates Classification_Data/ folder
   ```

3. **Rename Files:**
   ```bash
   python rename_dataset.py
   # Standardizes naming
   ```

4. **Train Model:**
   ```bash
   python train.py
   # Wait ~1 hour (GPU) or ~4-5 hours (CPU)
   ```

5. **Run Inference:**
   ```bash
   # Batch processing
   python file_inference.py -test_images/

   # Live camera
   python live_inference.py
   ```

---

## User Interface

**Location:** `src/ui/`

### Purpose
Tkinter-based GUI application for easy model deployment without command-line knowledge.

### Launch
```bash
cd src/ui
python app_main.py
```

### Architecture

**Framework:** Tkinter (Python's built-in GUI)  
**Design:** Dark theme with neon green accents  
**Window:** 1200×800, resizable (min 1100×700)  
**Pattern:** Screen-based navigation with state management

**Structure:**
```
ui/
├── app_main.py           # Main controller
├── components/           # Reusable UI elements
│   ├── styles.py         # Design tokens (colors, fonts)
│   └── widgets.py        # Custom components
├── screens/              # Screen modules
│   ├── home.py           # Mode & model selection
│   ├── input_select.py   # Source selection
│   ├── live_camera.py    # Live camera screen
│   ├── processing.py     # Batch processing
│   └── results.py        # Results viewer
└── utils/
    └── handlers.py       # Business logic
```

---

### Core Components

#### `app_main.py` - Application Controller

**Purpose:** Main window and navigation controller.

**Features:**
- Window management and sizing
- Navigation bar with Home/Input/Results buttons
- State management (mode, model, sources, output)
- Screen routing and transitions
- Navigation state control (enable/disable buttons)

**State Object:**
```python
state = {
    "mode": None,              # "classification" or "detection"
    "model_path": None,        # Path to best.pt
    "sources": None,           # List of file/folder paths
    "input_type": None,        # "file", "folder", or "live"
    "output_dir": None         # Results directory
}
```

**Navigation Flow:**
```
Home → Input Select → (Processing OR Live Camera) → Results
  ↑________↓____________↓_________________________↓
         Back navigation enabled throughout
```

**Navigation Bar:**
- **🏠 Home:** Always enabled
- **📂 Input:** Enabled after mode/model selection
- **📊 Results:** Enabled after processing complete

---

#### `components/styles.py` - Design System

**Theme:** Dark cyberpunk with neon green accents

**Color Palette:**
```python
COLORS = {
    "primary": "#00FF87",          # Neon green
    "bg_darkest": "#0A0E1A",       # Main background
    "bg_dark": "#141927",
    "bg_darker": "#1A1F35",        # Card background
    "bg_medium": "#242B42",        # Elevated elements
    "text_primary": "#E8EBF7",     # Light text
    "text_secondary": "#9BA3C1",   # Muted text
    "border": "#2A3148",           # Subtle border
    "success": "#00FF87",          # Same as primary
    "warning": "#FFD600",          # Bright yellow
    "danger": "#FF3D71",           # Bright red
}
```

**Typography:**
```python
FONTS = {
    "base": "Segoe UI",
    "heading": "Segoe UI Semibold",
    "mono": "Consolas"
}
```

**Spacing:** 4px, 8px, 12px, 16px, 24px, 32px  
**Shadows:** Layered frames for depth effect

---

#### `components/widgets.py` - Reusable Components

**Card:** Elevated surface with triple-layer shadow
```python
card = Card(parent, shadow="md", glow=True)
# Creates modern card with depth
```

**Button:** Neon glow on hover
```python
btn = Button(parent, text="Click Me", variant="primary", size="lg")
# Variants: primary, success, danger, dark, warning
# Sizes: sm, md, lg
```

**Label:** Styled text
```python
lbl = label(parent, "Text", variant="neon", weight="bold", size=14)
# Variants: primary, secondary, muted, neon, success, danger
```

**Heading:** Section headers
```python
heading(parent, "Title", level=2)
# Levels: 1, 2, 3
```

**ScrollFrame:** Scrollable container
```python
scroll = ScrollFrame(parent)
# Access inner frame: scroll.inner
```

**Divider:** Horizontal separator
```python
divider(parent)
```

---

### Screens

#### 1. `home.py` - Welcome & Configuration

**Purpose:** Select task mode and trained model.

**Features:**
- Mode selection (Classification vs Detection) with descriptions
- Model dropdown auto-populated from `runs/` folder
- Model discovery and validation
- Dynamic UI based on available models

**Model Discovery:**
```python
# Scans for trained models:
runs/classify/*/weights/best.pt  → Classification models
runs/detect/*/weights/best.pt    → Detection models
```

**User Flow:**
1. Select mode (radio buttons)
2. Choose model from dropdown
3. Click "Continue →"

**Next:** Input Select Screen

---

#### 2. `input_select.py` - Source Selection

**Purpose:** Choose what media to process.

**Three Input Options:**

**📄 Select Files:**
- Browse multiple images/videos
- Multi-select file dialog
- Shows count of selected files

**📁 Select Folder:**
- Browse directory
- Processes all images/videos recursively
- Shows folder name

**🔹 Live Camera:**
- Real-time inference
- No file selection needed
- Immediately starts processing

**Supported Formats:**
- Images: .jpg, .jpeg, .png, .bmp, .webp
- Videos: .mp4, .avi, .mov, .mkv

**User Flow:**
1. Click one of three options
2. Select files/folder (or skip for live)
3. Click "Start Processing →"

**Next:** Processing Screen OR Live Camera Screen

---

#### 3. `processing.py` - Batch Processing

**Purpose:** Process files/folders with visual feedback.

**Features:**
- Threaded processing (non-blocking UI)
- Progress bar with percentage
- Status text updates ("Processing image.jpg...")
- Cancel capability
- Neon green progress styling

**Processing Steps:**
1. Load model
2. Expand sources (scan folders for images/videos)
3. Process images sequentially
4. Process videos sequentially
5. Save results to `runs/ui_results/{mode}/{timestamp}/`

**Progress Updates:**
- Current file being processed
- Percentage complete
- Processing speed feedback

**Output Directory:**
```
runs/ui_results/
├── classification/
│   └── 20251107_143022/     # Timestamp
│       ├── image1.jpg
│       ├── image2.jpg
│       └── video_processed.mp4
└── detection/
    └── 20251107_143156/
        └── ...
```

**User Flow:**
1. Watch progress bar
2. Wait for completion (or cancel)
3. Auto-navigate to Results

**Next:** Results Screen

---

#### 4. `live_camera.py` - Real-time Inference

**Purpose:** Live camera feed with instant predictions.

**Architecture:**
- **Separate OpenCV window** (not embedded in Tkinter)
- **Threading:** Background initialization
- **Camera index:** 1 (external webcam)
- **Resolution:** 1280×720

**Features:**

**Classification Mode:**
- Top-1 prediction overlay
- Large text with confidence
- Simple, clean display

**Detection Mode:**
- Bounding boxes with labels
- Multi-object detection
- Color-coded classes

**Controls:**
- **Q (in camera window):** Stop stream
- **Back button (UI):** Cancel and return

**Why Separate Window:**
- OpenCV rendering = much faster than Tkinter canvas
- ~30 FPS vs ~10 FPS
- Lower latency
- Better user experience

**User Flow:**
1. Camera opens automatically
2. See predictions in real-time
3. Press Q to stop
4. Auto-return to Input Select

**Next:** Returns to Input Select Screen

---

#### 5. `results.py` - View & Export

**Purpose:** Display processed images/videos in thumbnail grid.

**Features:**

**Thumbnail Grid:**
- 3 columns, scrollable
- 260px wide thumbnails
- Hover glow effect (neon border)
- Filename labels

**Image Handling:**
- Click thumbnail for full-size modal
- High-quality rendering
- Preserve aspect ratio
- Smooth scaling

**Video Handling:**
- 🎬 placeholder icon
- Click to open in default player
- System file association

**Export:**
- "💾 Save Results" button
- Copy all to chosen folder
- Preserves filenames

**Action Buttons:**
- **← Back:** Return to Input Select
- **🏠 New Task:** Return to Home
- **💾 Save Results:** Export files

**Modal Viewer:**
- Full-size image display
- Close button (✕)
- Centered on screen
- Dark background

**User Flow:**
1. Browse thumbnail grid
2. Click images to view full-size
3. Click videos to open externally
4. Save results to folder
5. Start new task or go back

---

### Business Logic (`utils/handlers.py`)

**Core Functions:**

#### `discover_models()`
Scans `runs/` directory for trained models.

**Returns:**
```python
{
    "classification": [
        ("yolov8n_cls_V4", Path("runs/classify/yolov8n_cls_V4/weights/best.pt")),
        ("yolov8n_cls_V3", Path("runs/classify/yolov8n_cls_V3/weights/best.pt"))
    ],
    "detection": [
        ("yolov8n_detect_V5", Path("runs/detect/yolov8n_detect_V5/weights/best.pt"))
    ]
}
```

---

#### `expand_sources(sources)`
Converts file/folder selections into lists of images and videos.

**Input:** `[Path("folder/"), Path("image.jpg")]`  
**Output:** `([image1.jpg, image2.jpg, ...], [video1.mp4, ...])`

---

#### `run_inference(mode, model_path, sources, input_type, progress_callback)`
Main processing engine.

**Process:**
1. Load YOLO model
2. Expand sources into file lists
3. Create output directory with timestamp
4. Process images (PIL for classification, YOLO plot for detection)
5. Process videos (frame-by-frame with OpenCV)
6. Save annotated outputs
7. Return output directory path

**Progress Callback:**
```python
progress_callback(current, total, status_text)
# Updates UI progress bar
```

---

#### Classification Processing

**`process_classification_image()`**
- Load image with PIL
- Run YOLO inference
- Get top-1 prediction
- Draw label with PIL (high quality)
- Save to output

**`process_classification_video()`**
- Open video with OpenCV
- Process frame-by-frame
- Draw label with OpenCV (faster)
- Write to output video

**Configuration:**
- Font size: 50pt
- Confidence threshold: 0.3
- Text color: White
- Background: Semi-transparent black

---

#### Detection Processing

**`process_detection_image()`**
- Run YOLO inference
- Use YOLO's built-in plot() method
- Draws bounding boxes automatically
- Save to output

**`process_detection_video()`**
- Open video with OpenCV
- Process frame-by-frame
- Use YOLO's plot() on each frame
- Write to output video

**YOLO's plot() includes:**
- Bounding boxes
- Class labels
- Confidence scores
- Color-coded by class

---

#### `list_result_images(mode, result_dir)`
Scans output directory and returns list of all processed files.

**Returns:** `[Path("image1.jpg"), Path("image2.jpg"), Path("video.mp4"), ...]`

---

#### `copy_results_to(dest_dir, files)`
Copies result files to user-chosen destination.

**Uses:** `shutil.copy2()` to preserve metadata

---

### UI Workflow Examples

#### Classification Workflow:
```
1. Launch: python app_main.py
2. Home: Select "Classification" + "yolov8n_cls_V4"
3. Input: Select folder with images
4. Processing: Watch progress (2 min for 50 images)
5. Results: View thumbnails, click to enlarge
6. Export: Save to Desktop/results/
```

#### Detection Workflow:
```
1. Home: Select "Detection" + "yolov8n_detect_V5"
2. Input: Select folder with images
3. Processing: Watch progress (3 min for 50 images)
4. Results: View bounding boxes
5. Export: Save annotated images
```

#### Live Camera Workflow:
```
1. Home: Select mode + model
2. Input: Click "Start Camera"
3. Live: OpenCV window opens
4. Inference: See real-time predictions
5. Stop: Press Q
6. Return: Back to Input Select
```

---

### Design Highlights

✅ **Dark theme** with neon green accents  
✅ **Card-based layout** with depth shadows  
✅ **Hover effects** on interactive elements  
✅ **Progress feedback** for operations  
✅ **Threading** prevents UI freezing  
✅ **OpenCV window** for live camera (performance)  
✅ **Error handling** with user-friendly messages  
✅ **State persistence** across navigation  
✅ **Auto-discovery** of trained models  
✅ **Thumbnail grid** with full-size preview  

---

## Complete Workflows

### Full Pipeline: From Video to Deployment

#### Step 1: Data Collection (20 minutes)
```bash
cd src/data_collection
python data_collection.py
# Record 2 videos per class (20 seconds each)
# 10 classes × 2 videos × 1 minute = 20 minutes
```

**Output:** `Data_Video/<class>/*.avi` (20 videos total)

---

#### Step 2: Frame Extraction (7 minutes)
```bash
python extract_frames.py
# Converts videos to images
# ~1,200 frames per class
```

**Output:** `images/<class>/*.jpg` (~12,000 images)

---

#### Step 3: Auto-Annotation (20 minutes on RTX 5080)
```bash
cd ../data_annotation
python Auto_Annotate.py
# GPU-accelerated annotation
# 26,000 images in 20 minutes
```

**Output:** `Data/` with YOLO labels (26,000 annotated images)

---

#### Step 4: Verify Annotations (5 minutes)
```bash
python visualize_yolo.py
# Visual quality check
# Browse visualizations/ folder
```

---

#### Step 5: Detection Training (2 hours GPU / 8-10 hours CPU)
```bash
cd ../detection
python clean_invalid_labels.py
python dataset_stats.py
python train.py
# Train YOLOv8n detection model
```

**Output:** `runs/detect/yolov8n_detect_V5/weights/best.pt`

**Performance:** mAP50 ~98%, mAP50-95 ~96%

---

#### Step 6: Classification Training (1 hour GPU / 4-5 hours CPU)
```bash
cd ../classification
python organise_dataset.py
python rename_dataset.py
python train.py
# Train YOLOv8n-cls model
```

**Output:** `runs/classify/yolov8n_cls_V4/weights/best.pt`

**Performance:** Accuracy ~99.9%

---

#### Step 7: Deploy via UI (Instant)
```bash
cd ../ui
python app_main.py
# Use trained models with GUI
# No coding required
```

---

### Time Investment Summary

**Data Collection:** 20 minutes  
**Frame Extraction:** 7 minutes  
**Auto-Annotation:** 20 minutes (RTX 5080)  
**Verification:** 5 minutes  
**Detection Training:** 2 hours (GPU) or 8-10 hours (CPU)  
**Classification Training:** 1 hour (GPU) or 4-5 hours (CPU)  

**Total (GPU path):** ~4 hours active work  
**Total (CPU path):** ~13-16 hours (mostly waiting)

---

### Command-Line Quick Reference

#### Data Collection:
```bash
cd src/data_collection
python data_collection.py              # Record videos
python extract_frames.py               # Extract frames
```

#### Auto-Annotation:
```bash
cd src/data_annotation
python Auto_Annotate.py                # Auto-label dataset
python visualize_yolo.py               # Verify labels
```

#### Detection:
```bash
cd src/detection
python check_labels.py                 # Validate labels
python clean_invalid_labels.py         # Clean dataset
python dataset_stats.py                # View statistics
python train.py                        # Train detection model
python File_Inference.py -<path>       # Batch inference
python Live_Feed.py                    # Live camera
```

#### Classification:
```bash
cd src/classification
python organise_dataset.py             # Prepare dataset
python rename_dataset.py               # Rename files
python train.py                        # Train classification model
python file_inference.py -<path>       # Batch inference
python live_inference.py               # Live camera
```

#### User Interface:
```bash
cd src/ui
python app_main.py                     # Launch GUI
```

---

## Key Differences: Detection vs Classification

| Aspect | Detection | Classification |
|--------|-----------|----------------|
| **Task** | Localize objects (where + what) | Identify entire image (what only) |
| **Output** | Bounding boxes + class per object | Single class for whole image |
| **Data format** | images/ + labels/ folders | class-named folders only |
| **Label files** | Required (.txt with coordinates) | Not needed (folder = label) |
| **Image size** | 640×640 | 224×224 |
| **Training epochs** | 50 | 25 |
| **Training time** | Longer | Shorter |
| **Use case** | Multiple objects in scene | One dominant object |
| **Complexity** | Higher (localization + classification) | Lower (classification only) |
| **Model suffix** | `.pt` (yolov8n.pt) | `-cls.pt` (yolov8n-cls.pt) |

---

## Troubleshooting

### Common Issues

#### CUDA Errors (RTX 40XX/50XX)
**Problem:** PyTorch doesn't support latest GPU architectures  
**Solution:** Force CPU mode
```bash
# Windows
$env:CUDA_VISIBLE_DEVICES="-1"

# Linux/Mac
export CUDA_VISIBLE_DEVICES="-1"
```

Or add to script:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

---

#### Camera Not Opening
**Problem:** Camera index incorrect  
**Solution:** Try different camera indices

In code, change:
```python
cap = cv2.VideoCapture(1)  # Try 0, 1, or 2
```

---

#### Empty Annotations
**Problem:** Grounding DINO not detecting objects  
**Solutions:**
- Use more descriptive prompts: "black computer mouse" not "mouse"
- Check image quality (not too blurry/small)
- Verify GPU is being used
- Try different confidence thresholds

---

#### Model Not Found in UI
**Problem:** Model doesn't appear in dropdown  
**Solutions:**
- Ensure model is in correct location:
  - `runs/classify/*/weights/best.pt` for classification
  - `runs/detect/*/weights/best.pt` for detection
- Check folder structure matches expected pattern
- Restart UI application

---

#### Slow Training
**Problem:** Training takes too long  
**Solutions:**
- Use GPU instead of CPU (10× faster)
- Reduce batch size if running out of memory
- Use smaller model (YOLOv8n instead of YOLOv8s)
- Reduce number of epochs
- Enable AMP (automatic mixed precision)

---

#### Out of Memory
**Problem:** CUDA out of memory during training  
**Solutions:**
- Reduce batch size (32 → 16 → 8)
- Use smaller model (YOLOv8n)
- Close other GPU-using applications
- Enable gradient checkpointing (advanced)

---

## Performance Benchmarks

### RTX 5080 GPU

**Auto-Annotation:**
- 26,000 images in 20 minutes
- ~1,300 images/minute
- 750× faster than manual

**Detection Training:**
- 50 epochs in ~1.2 hours
- Final mAP50: ~98%

**Classification Training:**
- 25 epochs in ~0.6 hours
- Final accuracy: ~99.9%

**Inference:**
- Detection: ~50 FPS (live camera)
- Classification: ~100 FPS (live camera)

---

### CPU (Intel Core Ultra 7 265K, 20 cores)

**Detection Training:**
- 50 epochs in ~8-10 hours
- Final mAP50: ~98%

**Classification Training:**
- 25 epochs in ~4-5 hours
- Final accuracy: ~99.9%

**Inference:**
- Detection: ~15 FPS (live camera)
- Classification: ~30 FPS (live camera)

---

## Best Practices

### Data Collection
- Record in good lighting conditions
- Vary angles and perspectives
- Include different backgrounds
- Capture object in different states
- 20 seconds per video is optimal

### Training
- Always validate labels before training
- Use clean_invalid_labels.py to remove bad data
- Monitor training curves for overfitting
- Save best model, not last
- Use early stopping (patience=10)

### Inference
- Use appropriate confidence thresholds
- For live camera, use OpenCV window (not Tkinter)
- Batch process when possible
- Use GPU for large batches

### UI Usage
- Organize files before processing
- Use folders for batch processing
- Check model selection carefully
- Save results immediately after processing

---

## Future Enhancements

### Potential Improvements

**Data Pipeline:**
- Active learning to identify challenging examples
- Synthetic data generation for rare views
- Cross-environment testing
- Temporal augmentation

**Models:**
- Model ensemble for higher accuracy
- Quantization for mobile deployment
- Model distillation for edge devices
- Few-shot learning for new classes

**Infrastructure:**
- Cloud training pipeline
- A/B testing framework
- Automated retraining triggers
- Continuous data collection

**UI:**
- Mobile app (iOS/Android)
- Web-based interface
- Batch processing queue
- Model comparison tools

---

## Conclusion

The `src/` folder provides a complete, Deployment-ready pipeline for office item classification and detection. The modular architecture allows for:

✅ Easy data collection and annotation  
✅ Flexible model training (detection or classification)  
✅ Multiple inference methods (batch, live, UI)  
✅ Professional dataset organization  
✅ Comprehensive quality control tools  
✅ User-friendly GUI deployment  

**Total Development Time:** From zero to deployed model in ~4 hours (with GPU)

**Key Innovation:** Auto-annotation system reduces manual labeling from 250+ hours to 20 minutes (750× speedup)

---

**Author:** Oluwatunmise Shuaibu Raphael  
**Student ID:** M00960413  
**Module:** PDE3802 - AI in Robotics  
**Institution:** Middlesex University London  

*Last Updated: November 7, 2025*  
*Documentation Version: 1.0*