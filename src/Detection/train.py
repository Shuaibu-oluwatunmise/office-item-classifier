#Detection/train.py
import torch
from ultralytics import YOLO
from pathlib import Path


def main():
    # === CONFIGURATION ===
    DATA_YAML = Path("Data/data.yaml")
    MODEL_NAME = "yolov8n.pt"
    EPOCHS = 30
    BATCH_SIZE = 16
    IMG_SIZE = 640
    PATIENCE = 20

    # === DEVICE SELECTION ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training YOLOv8n Version 3 for {EPOCHS} epochs on device: {device.upper()}")

    # === FIX WINDOWS MULTIPROCESSING ===
    torch.multiprocessing.set_sharing_strategy("file_system")

    # === VERIFY DATA STRUCTURE ===
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"❌ Could not find dataset file at {DATA_YAML}")

    # === INITIALIZE MODEL ===
    model = YOLO(MODEL_NAME)

    # === TRAIN ===
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project="runs/detect",
        name="yolov8n_detect_V3",
        patience=PATIENCE,
        optimizer="AdamW",
        lr0=0.008,
        momentum=0.937,
        weight_decay=0.0005,
        pretrained=True,
        deterministic=True,
        cos_lr=False,
        amp=True,
        cache=False,
        workers=12,
        plots=True,
        verbose=True,

        # === AUGMENTATION SETTINGS ===
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.4,
        translate=0.15,
        scale=0.55,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.4,
        auto_augment="randaugment",

        # === DETECTION TASK OPTIONS ===
        overlap_mask=True,
        single_cls=False,
        val=True,
    )

    print("\n✅ Training complete! Check results under 'runs/detect/yolov8n_detect_V3'")


if __name__ == "__main__":
    main()
