import os
from datetime import datetime

BASE_PATH = "Data"
SUBSETS = ["train", "val", "test"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
LOG_FILE = "cleanup_log.txt"

def log_action(message):
    """Write a message to the log file and print it to console."""
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(message + "\n")
    print(message)

def clean_labels():
    total_deleted_labels = 0
    total_deleted_images = 0
    total_fixed_files = 0

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 60 + "\n")
        log.write(f"🧹 Cleanup started: {datetime.now()}\n")
        log.write("=" * 60 + "\n")

    for subset in SUBSETS:
        label_dir = os.path.join(BASE_PATH, subset, "labels")
        image_dir = os.path.join(BASE_PATH, subset, "images")

        if not os.path.exists(label_dir):
            log_action(f"[WARN] Labels folder not found: {label_dir}")
            continue

        for filename in os.listdir(label_dir):
            if not filename.endswith(".txt"):
                continue

            label_path = os.path.join(label_dir, filename)
            with open(label_path, "r") as f:
                lines = f.readlines()

            valid_lines = [line for line in lines if len(line.strip().split()) == 5]

            if not valid_lines:
                os.remove(label_path)
                total_deleted_labels += 1
                log_action(f"[DEL] Removed label: {label_path}")

                image_name = os.path.splitext(filename)[0]
                image_deleted = False
                for ext in IMAGE_EXTENSIONS:
                    image_path = os.path.join(image_dir, image_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        total_deleted_images += 1
                        log_action(f"     └─ Removed image: {image_path}")
                        image_deleted = True
                        break

                if not image_deleted:
                    log_action(f"     └─ No matching image found for: {image_name}")
            else:
                if len(valid_lines) < len(lines):
                    with open(label_path, "w") as f:
                        f.writelines(valid_lines)
                    total_fixed_files += 1
                    log_action(f"[FIX] Cleaned bad lines in: {label_path}")

    # Summary section
    summary = (
        "\n📊 SUMMARY\n"
        f"   • Labels deleted: {total_deleted_labels}\n"
        f"   • Images deleted: {total_deleted_images}\n"
        f"   • Labels cleaned: {total_fixed_files}\n"
    )
    log_action(summary)
    log_action("✅ Cleanup complete!\n")

if __name__ == "__main__":
    clean_labels()
