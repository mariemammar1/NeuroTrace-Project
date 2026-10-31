
from pathlib import Path
from collections import defaultdict
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = PROJECT_ROOT / "docs" / "dataset_report.csv"

OUTPUT_PATH = PROJECT_ROOT / "docs" / "duplicate_report.csv"

groups = defaultdict(list)

# Read the previous inspection report
with open(REPORT_PATH, newline="", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:
        groups[row["sha256"]].append(row)

# Keep only duplicate groups
duplicates = {
    file_hash: images
    for file_hash, images in groups.items()
    if len(images) > 1
}

cross_split = []
cross_class = []

duplicate_file_count = 0

for file_hash, images in duplicates.items():

    duplicate_file_count += len(images) - 1

    splits = {image["split"] for image in images}
    classes = {image["class"] for image in images}

    if len(splits) > 1:
        cross_split.append((file_hash, images))

    if len(classes) > 1:
        cross_class.append((file_hash, images))

print("=" * 60)
print("NEUROTRACE - DUPLICATE ANALYSIS")
print("=" * 60)

print(f"\nDuplicate groups: {len(duplicates)}")
print(f"Extra duplicate files: {duplicate_file_count}")

print(f"\nGroups crossing Training/Testing: {len(cross_split)}")

print(f"Groups appearing in multiple classes: {len(cross_class)}")

print("\nDuplicate groups by split and class:")

for file_hash, images in duplicates.items():

    locations = defaultdict(list)

    for image in images:
        key = (image["split"], image["class"])
        locations[key].append(image["filename"])

    print(f"\nHash: {file_hash[:16]}...")

    for (split, class_name), filenames in locations.items():
        print(f"  {split} / {class_name}: {len(filenames)} files")

        for filename in filenames:
            print(f"    {filename}")

# Save a detailed duplicate report
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:

    fieldnames = [
        "sha256",
        "split",
        "class",
        "filename"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for file_hash, images in duplicates.items():

        for image in images:

            writer.writerow({
                "sha256": file_hash,
                "split": image["split"],
                "class": image["class"],
                "filename": image["filename"]
            })

print("\nDetailed report saved to:")
print(OUTPUT_PATH)

print("\nDuplicate analysis completed.")
