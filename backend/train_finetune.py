
from pathlib import Path

import torch
import torch.nn as nn

from data_loader import train_loader, validation_loader
from model import build_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"

BEST_MODEL_PATH = (
    PROJECT_ROOT / "models" / "resnet18_finetuned_best.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10

LAYER4_LR = 0.00001
FC_LR = 0.0001

WEIGHT_DECAY = 0.0001


# ============================================================
# 2. LOAD BASELINE AND PREPARE FINE-TUNING MODEL
# ============================================================

def prepare_model():

    print("Loading baseline model...")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_MODEL_PATH}"
        )

    model = build_model(fine_tune=True)

    checkpoint = torch.load(
        BASELINE_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)

    print("Baseline weights loaded successfully.")
    print("Baseline best epoch:", checkpoint["epoch"])
    print(
        "Baseline validation accuracy:",
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    return model


# ============================================================
# 3. OPTIMIZER
# ============================================================

def build_optimizer(model):

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": LAYER4_LR
            },
            {
                "params": model.fc.parameters(),
                "lr": FC_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    return optimizer


# ============================================================
# 4. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, criterion, optimizer):

    model.train()

    # Keep frozen layers and their BatchNorm statistics unchanged.
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 5. VALIDATION
# ============================================================

def validate(model, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 6. SAVE BEST MODEL
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_accuracy
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": validation_accuracy * 100,
            "class_to_idx": train_loader.dataset.class_to_idx,
            "architecture": "ResNet18",
            "fine_tuning": True,
            "layer4_learning_rate": LAYER4_LR,
            "fc_learning_rate": FC_LR
        },
        BEST_MODEL_PATH
    )

    print("Best fine-tuned model saved!")


# ============================================================
# 7. MAIN TRAINING LOOP
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEUROTRACE - RESNET18 FINE-TUNING")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Layer4 learning rate:", LAYER4_LR)
    print("FC learning rate:", FC_LR)
    print("Batch size:", train_loader.batch_size)

    model = prepare_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model)

    best_validation_accuracy = 0.0

    for epoch in range(EPOCHS):

        print("\n" + "-" * 60)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print("-" * 60)

        train_loss, train_accuracy = train_one_epoch(
            model,
            criterion,
            optimizer
        )

        validation_loss, validation_accuracy = validate(
            model,
            criterion
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                validation_accuracy
            )

        else:
            print("No improvement in validation accuracy.")

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETED")
    print("=" * 60)

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print("Model saved at:", BEST_MODEL_PATH)
