from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ultralytics
from ultralytics import YOLO

BASE = Path.cwd()
RUNS = BASE / "runs"
UI_RESULTS = RUNS / "ui_results"
UI_RESULTS.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

# Classification overlay settings
FONT_SIZE = 50
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 0, 180)
CONF_THRESHOLD = 0.3

def discover_models() -> Dict[str, List[Tuple[str, Path]]]:
    """
    Returns:
      {
        "classification": [(run_name, path_to_best), ...],
        "detection": [(run_name, path_to_best), ...]
      }
    """
    out = {"classification": [], "detection": []}
    classify_root = RUNS / "classify"
    detect_root = RUNS / "detect"

    if classify_root.exists():
        for p in (classify_root.glob("*/weights/best.pt")):
            out["classification"].append((p.parents[1].name, p))

    if detect_root.exists():
        for p in (detect_root.glob("*/weights/best.pt")):
            out["detection"].append((p.parents[1].name, p))

    # Sort for consistent dropdown order
    out["classification"].sort(key=lambda x: x[0])
    out["detection"].sort(key=lambda x: x[0])
    return out

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_iter(paths) -> List[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]

def expand_sources(sources) -> Tuple[List[Path], List[Path]]:
    """
    Expand file/dir selection into separate lists of images and videos.
    Returns: (image_files, video_files)
    """
    image_files: List[Path] = []
    video_files: List[Path] = []
    
    for src in _ensure_iter(sources):
        if src.is_dir():
            # Scan directory for images and videos
            for ext in IMAGE_EXTS:
                image_files.extend(src.rglob(f"*{ext}"))
            for ext in VIDEO_EXTS:
                video_files.extend(src.rglob(f"*{ext}"))
        elif src.suffix.lower() in IMAGE_EXTS:
            image_files.append(src)
        elif src.suffix.lower() in VIDEO_EXTS:
            video_files.append(src)
    
    # Deduplicate
    image_files = list(dict.fromkeys(image_files))
    video_files = list(dict.fromkeys(video_files))
    
    return image_files, video_files

def draw_classification_label_pil(image: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Draw classification label on PIL image (top-1 only)"""
    draw = ImageDraw.Draw(image, "RGBA")
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(FONT_SIZE * 0.4)
    draw.rectangle([(10, 10), (10 + text_w + pad, 10 + text_h + pad)], fill=BOX_COLOR)
    draw.text((15, 15), text, fill=TEXT_COLOR, font=font)
    return image

def draw_classification_label_cv2(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw classification label on OpenCV frame (top-1 only)"""
    # Draw background box
    cv2.rectangle(frame, (10, 10), (500, 80), (0, 0, 0), -1)
    # Draw text
    cv2.putText(frame, text, (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return frame

def process_classification_image(model: YOLO, image_path: Path, output_path: Path, font: ImageFont.FreeTypeFont):
    """Process single image for classification (top-1 only)"""
    results = model.predict(source=str(image_path), verbose=False)
    r = results[0]

    if r.probs.top1conf > CONF_THRESHOLD:
        class_name = model.names[r.probs.top1]
        conf = r.probs.top1conf.item() * 100
        label_text = f"{class_name} - {conf:.2f}%"
    else:
        label_text = "Low confidence"

    image = Image.open(image_path).convert("RGB")
    image = draw_classification_label_pil(image, label_text, font)
    image.save(output_path)

def process_classification_video(model: YOLO, video_path: Path, output_path: Path):
    """Process video for classification (top-1 only, frame by frame)"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        r = results[0]
        
        if r.probs.top1conf > CONF_THRESHOLD:
            class_name = model.names[r.probs.top1]
            conf = r.probs.top1conf.item() * 100
            label_text = f"{class_name} - {conf:.2f}%"
            frame = draw_classification_label_cv2(frame, label_text)

        out.write(frame)

    cap.release()
    out.release()

def process_detection_image(model: YOLO, image_path: Path, output_path: Path):
    """Process single image for detection (with bounding boxes)"""
    results = model.predict(source=str(image_path), verbose=False)
    res = results[0]
    
    # Use YOLO's built-in plot method for detection with boxes
    img_annot = res.plot()
    cv2.imwrite(str(output_path), img_annot)

def process_detection_video(model: YOLO, video_path: Path, output_path: Path):
    """Process video for detection (bounding boxes frame by frame)"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        res = results[0]
        
        # Draw bounding boxes
        frame_annot = res.plot()
        out.write(frame_annot)

    cap.release()
    out.release()

def run_inference(mode: str, model_path: Path, sources, input_type: str, progress_callback=None) -> Path:
    """
    Runs inference over given sources (files/folder) and saves results.
    
    Args:
        mode: "classification" or "detection"
        model_path: Path to model weights
        sources: List of file/folder paths
        input_type: "file", "folder", or "live"
        progress_callback: Optional callback(current, total, status_text)
    
    Returns:
        Created output directory path
    """
    if input_type == "live":
        # Live camera doesn't create output directory
        return None
    
    # Expand sources into images and videos
    image_files, video_files = expand_sources(sources)
    total_files = len(image_files) + len(video_files)
    
    if total_files == 0:
        raise ValueError("No valid image or video files found in the selected source(s)")
    
    # Create output directory
    out_dir = UI_RESULTS / mode / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(str(model_path))
    
    # Load font for classification
    if mode == "classification":
        try:
            font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()
    
    current = 0
    
    # Process images
    for img_path in image_files:
        current += 1
        if progress_callback:
            progress_callback(current, total_files, f"Processing {img_path.name}...")
        
        output_path = out_dir / img_path.name
        
        if mode == "classification":
            process_classification_image(model, img_path, output_path, font)
        else:  # detection
            process_detection_image(model, img_path, output_path)
    
    # Process videos (DON'T increment progress until done)
    for idx, vid_path in enumerate(video_files):
        if progress_callback:
            progress_callback(current, total_files, f"Processing video {vid_path.name}... (may take time)")
        
        output_path = out_dir / f"{vid_path.stem}_processed{vid_path.suffix}"
        
        if mode == "classification":
            process_classification_video(model, vid_path, output_path)
        else:  # detection
            process_detection_video(model, vid_path, output_path)
        
        # Now increment after video is done
        current += 1
        if progress_callback:
            progress_callback(current, total_files, f"Completed {vid_path.name}")

    return out_dir

def list_result_images(mode: str, result_dir: Path = None) -> List[Path]:
    """Return list of image paths from the latest (or given) result dir for mode."""
    base = UI_RESULTS / mode
    if result_dir is None:
        if not base.exists():
            return []
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if not subdirs:
            return []
        # Latest by name (timestamp)
        result_dir = sorted(subdirs)[-1]

    # Return both images and videos
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(sorted(result_dir.glob(f"*{ext}")))
    for ext in VIDEO_EXTS:
        files.extend(sorted(result_dir.glob(f"*{ext}")))
    
    return files

def copy_results_to(dest_dir: Path, files: List[Path]) -> None:
    """Copy result files to destination directory"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(str(f), str(dest_dir / f.name))