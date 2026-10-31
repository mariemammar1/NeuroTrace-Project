
from pathlib import Path
import csv
import shutil

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from data_loader import test_loader
from model import build_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "docs" / "misclassified"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Maximum number of images displayed in each contact sheet
MAX_IMAGES_PER_SHEET = 12


# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

def load_trained_model():

    print("Loading trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = build_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Best epoch:", checkpoint["epoch"])

    return model


# ============================================================
# 3. IDENTIFY MISCLASSIFIED IMAGES
# ============================================================

def find_errors(model):

    dataset = test_loader.dataset

    class_names = dataset.classes

    # ImageFolder stores each image path and its true label
    samples = dataset.samples

    errors = []

    image_index = 0

    print("\nAnalyzing test images...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = torch.max(
                probabilities,
                dim=1
            )

            predictions = predictions.cpu().tolist()
            confidences = confidences.cpu().tolist()
            labels = labels.tolist()

            for i in range(len(labels)):

                true_label = labels[i]
                predicted_label = predictions[i]

                image_path = samples[image_index][0]

                if true_label != predicted_label:

                    true_class = class_names[true_label]
                    predicted_class = class_names[predicted_label]

                    error = {
                        "image_path": image_path,
                        "filename": Path(image_path).name,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "confidence": confidences[i]
                    }

                    errors.append(error)

                image_index += 1

    print(f"\nTotal test images: {len(samples)}")
    print(f"Misclassified images: {len(errors)}")
    print(f"Correctly classified: {len(samples) - len(errors)}")

    return errors, class_names


# ============================================================
# 4. ORGANIZE MISCLASSIFIED IMAGES
# ============================================================

def organize_errors(errors):

    print("\nOrganizing misclassified images...")

    # Avoid keeping outdated copies from a previous execution
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for error in errors:

        folder_name = (
            f"{error['true_class']}_as_{error['predicted_class']}"
        )

        destination_dir = OUTPUT_DIR / folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / error["filename"]

        shutil.copy2(
            error["image_path"],
            destination_path
        )

        error["copied_path"] = str(destination_path)

    print("Images organized successfully.")


# ============================================================
# 5. SAVE CSV REPORT
# ============================================================

def save_error_report(errors):

    report_path = OUTPUT_DIR / "error_report.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "filename",
            "true_class",
            "predicted_class",
            "confidence",
            "image_path",
            "copied_path"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for error in errors:

            row = {
                "filename": error["filename"],
                "true_class": error["true_class"],
                "predicted_class": error["predicted_class"],
                "confidence": round(error["confidence"] * 100, 2),
                "image_path": error["image_path"],
                "copied_path": error["copied_path"]
            }

            writer.writerow(row)

    print(f"CSV report saved: {report_path}")


# ============================================================
# 6. GENERATE VISUAL CONTACT SHEETS
# ============================================================

def generate_contact_sheets(errors):

    print("\nGenerating contact sheets...")

    # Group errors by actual class and predicted class
    groups = {}

    for error in errors:

        key = (
            error["true_class"],
            error["predicted_class"]
        )

        groups.setdefault(key, []).append(error)

    for (true_class, predicted_class), group_errors in groups.items():

        selected_errors = group_errors[:MAX_IMAGES_PER_SHEET]

        number_of_images = len(selected_errors)

        columns = 4
        rows = (number_of_images + columns - 1) // columns

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(12, 3 * rows)
        )

        # Handle cases where there is only one row
        if rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        for index, error in enumerate(selected_errors):

            image = Image.open(error["image_path"])
            image = ImageOps.exif_transpose(image).convert("RGB")

            axes[index].imshow(image)

            axes[index].set_title(
                f"{error['filename']}\n"
                f"Confidence: {error['confidence'] * 100:.1f}%",
                fontsize=8
            )

            axes[index].axis("off")

        # Hide unused subplot positions
        for index in range(number_of_images, len(axes)):
            axes[index].axis("off")

        fig.suptitle(
            f"Actual: {true_class} | Predicted: {predicted_class}\n"
            f"Showing {number_of_images} of {len(group_errors)} errors",
            fontsize=13
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"{true_class}_as_{predicted_class}_contact_sheet.png"
        )

        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Saved: {output_path}")


# ============================================================
# 7. DISPLAY ERROR SUMMARY
# ============================================================

def display_error_summary(errors):

    print("\n" + "=" * 60)
    print("NEUROTRACE - MISCLASSIFICATION SUMMARY")
    print("=" * 60)

    groups = {}

    for error in errors:

        key = (
            error["true_class"],
            error["predicted_class"]
        )

        groups[key] = groups.get(key, 0) + 1

    for (true_class, predicted_class), count in sorted(groups.items()):

        print(
            f"{true_class:12} -> "
            f"{predicted_class:12} : {count}"
        )

    print("=" * 60)


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    model = load_trained_model()

    errors, class_names = find_errors(model)

    organize_errors(errors)

    save_error_report(errors)

    generate_contact_sheets(errors)

    display_error_summary(errors)

    print("\nERROR ANALYSIS COMPLETED")
    print("Results directory:", OUTPUT_DIR)
