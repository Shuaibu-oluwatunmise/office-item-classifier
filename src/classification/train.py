import torch
from ultralytics import YOLO
from pathlib import Path

def main():
    # === CONFIGURATION ===
    DATA_DIR = Path("Classification_Data")
    MODEL_NAME = "yolov8n-cls.pt"
    EPOCHS = 25
    IMG_SIZE = 224
    BATCH_SIZE = 32
    PATIENCE = 10

    # === DEVICE SELECTION ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training YOLOv8n-cls for {EPOCHS} epochs on device: {device.upper()}")

    # === FIX WINDOWS MULTIPROCESSING ===
    torch.multiprocessing.set_sharing_strategy("file_system")

    # === VERIFY DATA STRUCTURE ===
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("❌ Could not find 'train' or 'val' folders inside Classification_Data/")

    # === INITIALIZE MODEL ===
    model = YOLO(MODEL_NAME)

    # === TRAIN ===
    model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project="runs/classify",
        name="yolov8n_cls_V3",
        patience=PATIENCE,
        workers=12,
        verbose=True,
    )

    print("\n✅ Training complete! Check results under 'runs/classify/yolov8n_cls_V3'")


if __name__ == "__main__":
    main()
