from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import shutil
import glob
import cv2

# Ultralytics
from ultralytics import YOLO

BASE = Path.cwd()
RUNS = BASE / "runs"
UI_RESULTS = RUNS / "ui_results"
UI_RESULTS.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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

def expand_sources(sources) -> List[Path]:
    """Expand file/dir selection into a flat list of image files."""
    files: List[Path] = []
    for src in _ensure_iter(sources):
        if src.is_dir():
            for ext in IMAGE_EXTS:
                files.extend(src.rglob(f"*{ext}"))
        elif src.suffix.lower() in IMAGE_EXTS:
            files.append(src)
    # Deduplicate, keep relative order-ish
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def run_inference(mode: str, model_path: Path, sources) -> Path:
    """
    Runs inference over given sources and saves ONLY processed images
    into runs/ui_results/<mode>/<timestamp> preserving original filenames.
    Returns the created output directory.
    """
    files = expand_sources(sources)
    out_dir = UI_RESULTS / mode / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    for fp in files:
        # Predict single image
        results = model.predict(source=str(fp), verbose=False)
        res = results[0]

        # For detection/seg pose -> res.plot() returns annotated ndarray (BGR)
        # For classification -> res.plot() returns image with top-1 text overlay.
        img_annot = res.plot()
        save_path = out_dir / fp.name
        # cv2 expects BGR ndarray
        cv2.imwrite(str(save_path), img_annot)

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

    imgs: List[Path] = []
    for ext in IMAGE_EXTS:
        imgs.extend(sorted(result_dir.glob(f"*{ext}")))
    return imgs

def copy_results_to(dest_dir: Path, images: List[Path]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(str(img), str(dest_dir / img.name))
