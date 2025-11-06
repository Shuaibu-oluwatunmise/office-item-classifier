#classification/file_inference.py
import sys
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm  # ✅ for progress bars

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = Path("runs/classify/yolov8n_cls_V3/weights/best.pt")

FONT_SIZE = 60
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 0, 180)
CONF_THRESHOLD = 0.3
FONT_PATH = None

VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
VALID_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

# -------------------------------
# DRAW LABEL ON IMAGE (PIL)
# -------------------------------
def draw_label_pil(image, text, font):
    draw = ImageDraw.Draw(image, "RGBA")
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(FONT_SIZE * 0.4)
    draw.rectangle([(10, 10), (10 + text_w + pad, 10 + text_h + pad)], fill=BOX_COLOR)
    draw.text((15, 15), text, fill=TEXT_COLOR, font=font)
    return image

# -------------------------------
# MAIN
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python file_inference.py -<path_to_file_or_folder>")
    sys.exit(1)

input_path = Path(sys.argv[1].lstrip('-'))
if not input_path.exists():
    print(f"❌ Error: '{input_path}' not found.")
    sys.exit(1)

print(f"📦 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

try:
    font = ImageFont.truetype(FONT_PATH or "arial.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()

# ✅ always save results inside the given folder
results_dir = (input_path if input_path.is_dir() else input_path.parent) / "results"
results_dir.mkdir(exist_ok=True)

# -------------------------------
# FOLDER MODE
# -------------------------------
if input_path.is_dir():
    files = list(input_path.iterdir())
    image_files = [f for f in files if f.suffix.lower() in VALID_IMAGE_EXTS]
    video_files = [f for f in files if f.suffix.lower() in VALID_VIDEO_EXTS]

    if image_files:
        print(f"\n🖼️ Processing {len(image_files)} image(s)...")
        for f in tqdm(image_files, desc="Images", unit="img"):
            results = model.predict(source=f, verbose=False)
            r = results[0]

            if r.probs.top1conf > CONF_THRESHOLD:
                class_name = model.names[r.probs.top1]
                conf = r.probs.top1conf.item() * 100
                label_text = f"{class_name} - {conf:.2f}%"
            else:
                label_text = "Low confidence"

            image = Image.open(f).convert("RGB")
            image = draw_label_pil(image, label_text, font)
            image.save(results_dir / f.name)

    if video_files:
        print(f"\n🎞️ Processing {len(video_files)} video(s)...")
        for f in tqdm(video_files, desc="Videos", unit="vid"):
            cap = cv2.VideoCapture(str(f))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(3)), int(cap.get(4))
            out_path = results_dir / f"{f.stem}_classified.mp4"
            out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # ✅ frame-level progress bar
            for _ in tqdm(range(total_frames), desc=f"{f.name}", unit="frame", leave=False):
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, verbose=False)
                r = results[0]
                if r.probs.top1conf > CONF_THRESHOLD:
                    class_name = model.names[r.probs.top1]
                    conf = r.probs.top1conf.item() * 100
                    label_text = f"{class_name} - {conf:.2f}%"
                else:
                    label_text = ""

                if label_text:
                    cv2.rectangle(frame, (10, 10), (500, 80), (0, 0, 0), -1)
                    cv2.putText(frame, label_text, (25, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                out.write(frame)

            cap.release()
            out.release()

    print(f"\n✅ All results saved to: {results_dir}")

# -------------------------------
# SINGLE FILE MODE
# -------------------------------
elif input_path.suffix.lower() in VALID_IMAGE_EXTS.union(VALID_VIDEO_EXTS):
    suffix = input_path.suffix.lower()

    if suffix in VALID_IMAGE_EXTS:
        print("🖼️ Processing single image...")
        results = model.predict(source=input_path, verbose=False)
        r = results[0]
        if r.probs.top1conf > CONF_THRESHOLD:
            class_name = model.names[r.probs.top1]
            conf = r.probs.top1conf.item() * 100
            label_text = f"{class_name} - {conf:.2f}%"
        else:
            label_text = "Low confidence"

        image = Image.open(input_path).convert("RGB")
        image = draw_label_pil(image, label_text, font)
        image.save(results_dir / input_path.name)
        print(f"✅ Saved annotated image to: {results_dir}")

    else:
        print("🎞️ Processing single video...")
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(3)), int(cap.get(4))
        out_path = results_dir / f"{input_path.stem}_classified.mp4"
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for _ in tqdm(range(total_frames), desc=input_path.name, unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False)
            r = results[0]
            if r.probs.top1conf > CONF_THRESHOLD:
                class_name = model.names[r.probs.top1]
                conf = r.probs.top1conf.item() * 100
                label_text = f"{class_name} - {conf:.2f}%"
            else:
                label_text = ""

            if label_text:
                cv2.rectangle(frame, (10, 10), (500, 80), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (25, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            out.write(frame)

        cap.release()
        out.release()
        print(f"✅ Saved annotated video to: {out_path}")
else:
    print("⚠️ Unsupported file type. Please provide a folder, image, or video.")
