# src/ui/utils/handlers.py
import os, sys, time, subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO

# === Model paths from your project ===
CLS_MODEL = Path("runs/classify/yolov8n_cls_V3/weights/best.pt")
DET_MODEL = Path("runs/detect/yolov8n_detect_V2/weights/best.pt")

# fallbacks
def _font(size=28):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def _ensure_out(mode: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / "ui_results" / mode / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

def _iter_media(path: Path) -> Tuple[List[Path], List[Path]]:
    if path.is_file():
        return ([path], []) if path.suffix.lower() in IMAGE_EXTS else ([], [path])
    imgs = [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    vids = [p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    return imgs, vids

# ---------- Classification (file/folder) ----------
def classify_path(input_path: Path, conf: float = 0.30):
    model = YOLO(str(CLS_MODEL))
    out_dir = _ensure_out("classification")
    font = _font(40)

    imgs, vids = _iter_media(input_path)
    saved = []

    # Images
    for img_p in imgs:
        im = Image.open(img_p).convert("RGB")
        res = model.predict(im, verbose=False)[0]
        label = ""
        if res.probs is not None and float(res.probs.top1conf) >= conf:
            name = model.names[int(res.probs.top1)]
            label = f"{name}  {float(res.probs.top1conf)*100:.1f}%"
        draw = ImageDraw.Draw(im, "RGBA")
        if label:
            bbox = draw.textbbox((0,0), label, font=font)
            pad=12
            draw.rectangle([ (10,10), (10+bbox[2]-bbox[0]+pad, 10+bbox[3]-bbox[1]+pad) ],
                           fill=(0,0,0,180))
            draw.text((16,14), label, fill=(255,255,255), font=font)
        out_p = out_dir / img_p.name
        im.save(out_p)
        saved.append(out_p)

    # Videos (simple overlay on each frame)
    for vid_p in vids:
        cap = cv2.VideoCapture(str(vid_p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(3)), int(cap.get(4))
        out_p = out_dir / f"{vid_p.stem}_classified.mp4"
        writer = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, frame = cap.read()
            if not ok: break
            res = model.predict(frame, verbose=False)[0]
            if res.probs is not None and float(res.probs.top1conf) >= conf:
                name = model.names[int(res.probs.top1)]
                label = f"{name}  {float(res.probs.top1conf)*100:.1f}%"
                cv2.rectangle(frame, (10,10), (450,70), (0,0,0), -1)
                cv2.putText(frame, label, (20,55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            writer.write(frame)
        writer.release(); cap.release()
        saved.append(out_p)

    return out_dir, saved

# ---------- Detection (file/folder) ----------
def detect_path(input_path: Path, conf: float = 0.25, iou: float = 0.45):
    model = YOLO(str(DET_MODEL))
    out_dir = _ensure_out("detection")
    imgs, vids = _iter_media(input_path)
    saved = []

    # Images
    for img_p in imgs:
        im = cv2.imread(str(img_p))
        if im is None: continue
        res = model(im, conf=conf, iou=iou, verbose=False)[0]
        annotated = im.copy()
        names = model.names
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                cf = float(b.conf[0].cpu().numpy())
                color = (99,102,241)  # indigo-ish
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
                label = f"{names[cls]} {cf*100:.1f}%"
                (tw,th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1,y1-th-10), (x1+tw+10,y1), color, -1)
                cv2.putText(annotated, label, (x1+5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        out_p = out_dir / img_p.name
        cv2.imwrite(str(out_p), annotated)
        saved.append(out_p)

    # (Optional) videos – can be added similarly; keeping lean for reliability now
    return out_dir, saved

# ---------- Live feed ----------
def start_live_classification():
    # Spawns your existing live script
    script = Path("src/Classification/live_inference.py")
    return subprocess.Popen([sys.executable, str(script)])

def start_live_detection():
    # Spawns your existing threaded/recording app
    script = Path("src/Detection/Live_Feed.py")
    return subprocess.Popen([sys.executable, str(script)])
