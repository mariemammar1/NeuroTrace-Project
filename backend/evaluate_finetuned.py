
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data_loader import test_loader
from model import build_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT / "models" / "resnet18_finetuned_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "docs" / "finetuned_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. LOAD FINE-TUNED MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Build the same ResNet18 architecture
    model = build_model(fine_tune=True)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()

    print("Fine-tuned model loaded successfully.")
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Best validation accuracy:",
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    return model


# ============================================================
# 3. PREDICTIONS
# ============================================================

def get_predictions(model):

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions


# ============================================================
# 4. METRICS
# ============================================================

def evaluate(y_true, y_pred, class_names):

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("NEUROTRACE - FINE-TUNED TEST EVALUATION")
    print("=" * 60)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(report)

    return accuracy, report, matrix


# ============================================================
# 5. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(matrix, class_names):

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("NeuroTrace - Fine-Tuned Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "finetuned_confusion_matrix.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\nConfusion matrix saved:", output_path)


# ============================================================
# 6. SAVE REPORT
# ============================================================

def save_report(accuracy, report):

    output_path = OUTPUT_DIR / "finetuned_evaluation_report.txt"

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("NEUROTRACE - FINE-TUNED MODEL EVALUATION\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")

        file.write("Classification Report:\n")
        file.write(report)

    print("Report saved:", output_path)


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    print("Device:", DEVICE)
    print("Test images:", len(test_loader.dataset))

    class_names = test_loader.dataset.classes

    model = load_model()

    y_true, y_pred = get_predictions(model)

    accuracy, report, matrix = evaluate(
        y_true,
        y_pred,
        class_names
    )

    save_confusion_matrix(matrix, class_names)

    save_report(accuracy, report)

    print("\nFINE-TUNED EVALUATION COMPLETED")
