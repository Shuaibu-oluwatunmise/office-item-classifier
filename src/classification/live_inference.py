#classification/live_inference.py
import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "runs/classify/yolov8n_cls_V3/weights/best.pt"
FONT_SIZE = 36
FONT_PATH = None  # Optional custom .ttf font
CONF_THRESHOLD = 0.3  # minimum confidence to show prediction
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 0, 180)

# -------------------------------
# LOAD MODEL AND FONT
# -------------------------------
print(f"📦 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

try:
    font = ImageFont.truetype(FONT_PATH or "arial.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()

# -------------------------------
# OPEN CAMERA
# -------------------------------
cap = cv2.VideoCapture(1)  # use 0 for default camera

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("🎥 Live classification started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, verbose=False)
    r = results[0]

    if r.probs.top1conf > CONF_THRESHOLD:
        class_name = model.names[r.probs.top1]
        conf = r.probs.top1conf.item() * 100
        label_text = f"{class_name} - {conf:.2f}%"
    else:
        label_text = "Low confidence"

    # Convert frame (OpenCV → PIL) to draw nice text
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil, "RGBA")

    # Text box
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(FONT_SIZE * 0.4)
    draw.rectangle([(10, 10), (10 + text_w + pad, 10 + text_h + pad)], fill=BOX_COLOR)
    draw.text((15, 15), label_text, fill=TEXT_COLOR, font=font)

    # Convert back to OpenCV format for display
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Show frame
    cv2.imshow("Live Classification", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Stream ended.")
