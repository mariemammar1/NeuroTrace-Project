
from pathlib import Path
from collections import defaultdict
import hashlib
import shutil
import csv

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
CLEAN_DIR = PROJECT_ROOT / "dataset" / "cleaned"

REPORT_DIR = PROJECT_ROOT / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

SPLITS = ["Training", "Testing"]

# Create cleaned dataset
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

report = []

total_original = 0
total_kept = 0
total_removed = 0

print("=" * 60)
print("NEUROTRACE - DATASET CLEANING")
print("=" * 60)

for split in SPLITS:

    split_path = RAW_DIR / split

    if not split_path.exists():
        print(f"WARNING: Missing folder {split}")
        continue

    for class_folder in sorted(split_path.iterdir()):

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        output_folder = CLEAN_DIR / split / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Track hashes separately for each split and class
        seen_hashes = set()

        image_paths = sorted(
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        kept = 0
        removed = 0

        for image_path in image_paths:

            total_original += 1

            file_hash = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()

            if file_hash in seen_hashes:

                removed += 1
                total_removed += 1

                print(f"DUPLICATE SKIPPED: {image_path.name}")
                continue

            seen_hashes.add(file_hash)

            destination = output_folder / image_path.name

            shutil.copy2(image_path, destination)

            kept += 1
            total_kept += 1

        print(
            f"{split}/{class_name}: "
            f"{len(image_paths)} original, "
            f"{kept} kept, "
            f"{removed} removed"
        )

        report.append({
            "split": split,
            "class": class_name,
            "original": len(image_paths),
            "kept": kept,
            "removed": removed
        })

# Save report
report_path = REPORT_DIR / "cleaning_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split", "class", "original", "kept", "removed"
        ]
    )

    writer.writeheader()
    writer.writerows(report)

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print(f"Original images: {total_original}")
print(f"Images retained: {total_kept}")
print(f"Duplicates skipped: {total_removed}")

print(f"\nReport saved to: {report_path}")
print(f"Cleaned dataset: {CLEAN_DIR}")
