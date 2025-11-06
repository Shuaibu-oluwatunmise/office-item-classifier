#classification/rename_dataset.py
import os
from pathlib import Path

# Root of your classification dataset
ROOT = Path("Classification_Data")

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

for split in ["train", "val", "test"]:
    split_path = ROOT / split
    if not split_path.exists():
        continue

    for class_dir in split_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        )

        print(f"📂 Renaming {len(images)} images in '{split}/{class_name}'...")

        for idx, img in enumerate(images, start=1):
            new_name = f"{class_name}_{idx:04d}_{split}{img.suffix.lower()}"
            new_path = class_dir / new_name

            # Skip renaming if file already matches new pattern
            if img.name == new_name:
                continue

            img.rename(new_path)

print("\n✅ All images renamed successfully with split suffix (_train, _val, _test)!")
