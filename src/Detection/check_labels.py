#Detection/check_labels.py
import os

BASE_PATH = "Data"
SUBSETS = ["train", "val", "test"]

def check_label_files():
    invalid_files = []

    for subset in SUBSETS:
        label_dir = os.path.join(BASE_PATH, subset, "labels")
        if not os.path.exists(label_dir):
            print(f"[WARN] Labels folder not found: {label_dir}")
            continue

        for filename in os.listdir(label_dir):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(label_dir, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()

            # Check if any line has more than 5 values
            if any(len(line.strip().split()) > 5 for line in lines if line.strip()):
                invalid_files.append(filepath)

    if invalid_files:
        print("\n⚠️ Invalid label files found:")
        for f in invalid_files:
            print(f" - {f}")
    else:
        print("✅ All label files are valid!")

if __name__ == "__main__":
    check_label_files()
