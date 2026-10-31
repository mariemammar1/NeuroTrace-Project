
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, UnidentifiedImageError
import hashlib
import csv
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Dataset configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw"

SPLITS = ["Training", "Testing"]

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# --------------------------------------------------
# 2. Check dataset directory
# --------------------------------------------------

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_ROOT}"
    )

print("=" * 60)
print("NEUROTRACE - DATASET INSPECTION")
print("=" * 60)

print(f"\nDataset location: {DATASET_ROOT}")

# --------------------------------------------------
# 3. Inspect images
# --------------------------------------------------

records = []
class_counts = defaultdict(Counter)
dimensions = Counter()
formats = Counter()
hash_locations = defaultdict(list)

corrupted_images = []

for split in SPLITS:

    split_path = DATASET_ROOT / split

    if not split_path.exists():
        print(f"\nWARNING: {split} folder not found.")
        continue

    print(f"\n--- {split.upper()} ---")

    class_folders = sorted(
        folder for folder in split_path.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        print("No class folders found!")
        continue

    for class_folder in class_folders:

        class_name = class_folder.name

        image_paths = [
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        print(f"\nClass: {class_name}")
        print(f"Images found: {len(image_paths)}")

        for image_path in image_paths:

            try:
                # Read image properties
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = image.format
                    mode = image.mode

                # Calculate SHA-256 of original file
                file_hash = hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest()

                dimensions[(width, height)] += 1
                formats[image_format] += 1

                class_counts[split][class_name] += 1

                hash_locations[file_hash].append(
                    str(image_path)
                )

                records.append({
                    "split": split,
                    "class": class_name,
                    "filename": image_path.name,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "mode": mode,
                    "sha256": file_hash
                })

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ) as error:

                corrupted_images.append(str(image_path))

                print(f"Unreadable image: {image_path}")
                print(f"Reason: {error}")

# --------------------------------------------------
# 4. Display statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total readable images: {len(records)}")
print(f"Corrupted/unreadable images: {len(corrupted_images)}")

print("\nClass distribution:")

for split in SPLITS:
    print(f"\n{split}:")

    for class_name, count in sorted(
        class_counts[split].items()
    ):
        print(f"  {class_name}: {count}")

print("\nImage formats:")

for image_format, count in formats.items():
    print(f"  {image_format}: {count}")

print("\nMost common dimensions:")

for dimension, count in dimensions.most_common(10):
    print(f"  {dimension}: {count} images")

# --------------------------------------------------
# 5. Detect exact duplicate files
# --------------------------------------------------

duplicates = {
    file_hash: paths
    for file_hash, paths in hash_locations.items()
    if len(paths) > 1
}

print("\nExact duplicate groups:", len(duplicates))

for file_hash, paths in list(duplicates.items())[:10]:
    print(f"\nHash: {file_hash[:16]}...")

    for path in paths:
        print(f"  {path}")

# --------------------------------------------------
# 6. Save CSV report
# --------------------------------------------------

report_path = PROJECT_ROOT / "docs" / "dataset_report.csv"
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split", "class", "filename",
            "width", "height", "format",
            "mode", "sha256"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

print(f"\nCSV report saved to: {report_path}")

# --------------------------------------------------
# 7. Display sample images
# --------------------------------------------------

sample_records = []

for class_name in sorted({
    record["class"] for record in records
}):

    samples = [
        record for record in records
        if record["class"] == class_name
        and record["split"] == "Training"
    ]

    if samples:
        sample_records.append(samples[0])

if sample_records:

    fig, axes = plt.subplots(
        1,
        len(sample_records),
        figsize=(4 * len(sample_records), 4)
    )

    if len(sample_records) == 1:
        axes = [axes]

    for axis, record in zip(axes, sample_records):

        image_path = (
            DATASET_ROOT
            / record["split"]
            / record["class"]
            / record["filename"]
        )

        image = Image.open(image_path).convert("RGB")

        axis.imshow(image)
        axis.set_title(record["class"])
        axis.axis("off")

    plt.suptitle("NeuroTrace - MRI Dataset Samples")
    plt.tight_layout()
    plt.show()

print("\nDataset inspection completed.")
