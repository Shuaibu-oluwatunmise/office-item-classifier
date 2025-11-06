#Detection/dataset_stats.py
import os
from collections import Counter

BASE_PATH = "Data"
SUBSETS = ["train", "val", "test"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def count_dataset_stats():
    total_images = 0
    total_labels = 0
    total_instances = 0
    class_counter = Counter()

    print("\n📊 DATASET STATISTICS\n" + "=" * 60)

    for subset in SUBSETS:
        image_dir = os.path.join(BASE_PATH, subset, "images")
        label_dir = os.path.join(BASE_PATH, subset, "labels")

        image_count = sum(1 for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS) if os.path.exists(image_dir) else 0
        label_count = sum(1 for f in os.listdir(label_dir) if f.endswith(".txt")) if os.path.exists(label_dir) else 0

        total_images += image_count
        total_labels += label_count

        # Count instances per class
        subset_instances = 0
        if os.path.exists(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(label_dir, filename)
                    with open(filepath, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = parts[0]
                                class_counter[class_id] += 1
                                subset_instances += 1
        total_instances += subset_instances

        print(f"\n📁 {subset.upper()}")
        print(f"   • Images: {image_count}")
        print(f"   • Labels: {label_count}")
        print(f"   • Object instances: {subset_instances}")

    print("\n" + "=" * 60)
    print(f"📦 TOTAL IMAGES: {total_images}")
    print(f"📄 TOTAL LABEL FILES: {total_labels}")
    print(f"🧩 TOTAL OBJECT INSTANCES: {total_instances}")

    print("\n🔢 CLASS DISTRIBUTION:")
    for cls, count in sorted(class_counter.items(), key=lambda x: int(x[0])):
        print(f"   • Class {cls}: {count} instances")

    print("\n✅ Dataset statistics complete!\n")

if __name__ == "__main__":
    count_dataset_stats()
