#organise_dataset.py
import os
import random
import shutil
from pathlib import Path

# === CONFIGURATION ===
SOURCE_DIR = Path("Data")
DEST_DIR = Path("Classification_Data")
EXCESS_DIR = Path("excesses")

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
MAX_IMAGES_PER_CLASS = 1950

# === CREATE TARGET DIRECTORIES ===
for folder in [
    DEST_DIR / "train",
    DEST_DIR / "val",
    DEST_DIR / "test",
    EXCESS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# === SUPPORTED IMAGE EXTENSIONS ===
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# === MAIN LOGIC ===
for class_dir in SOURCE_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    images = [
        f for f in class_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    ]

    total = len(images)
    print(f"\n📂 Class '{class_name}': {total} images")

    if total == 0:
        print("   ⚠️ No images found, skipping.")
        continue

    # Shuffle images for random split
    random.shuffle(images)

    # Handle excess images
    if total > MAX_IMAGES_PER_CLASS:
        keep = images[:MAX_IMAGES_PER_CLASS]
        excess = images[MAX_IMAGES_PER_CLASS:]

        excess_target = EXCESS_DIR / class_name
        excess_target.mkdir(parents=True, exist_ok=True)

        for img in excess:
            shutil.move(str(img), excess_target / img.name)

        print(f"   🗑️ Moved {len(excess)} excess images to '{excess_target}'")
        images = keep
        total = len(images)

    # Determine split indices
    n_train = int(total * TRAIN_SPLIT)
    n_val = int(total * VAL_SPLIT)
    n_test = total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    # Create class subfolders and move images
    for split_name, split_imgs in splits.items():
        dest = DEST_DIR / split_name / class_name
        dest.mkdir(parents=True, exist_ok=True)

        for img in split_imgs:
            shutil.move(str(img), dest / img.name)

        print(f"   ✅ {split_name}: {len(split_imgs)} images")

print("\n✅ Dataset organization complete!")
