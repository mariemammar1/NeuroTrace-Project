
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import (
    train_loader,
    validation_loader
)

from model import build_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 10
LEARNING_RATE = 0.0001

BEST_MODEL_PATH = MODEL_DIR / "resnet18_best.pth"


# ==========================================
# 2. INITIALISATION
# ==========================================

model = build_model()
model = model.to(DEVICE)

# Fonction de perte
criterion = nn.CrossEntropyLoss()

# Optimiser uniquement la dernière couche
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 3. FONCTION D'ENTRAINEMENT
# ==========================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Réinitialiser les gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculer la perte
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Mettre à jour les poids
        optimizer.step()

        # Accumuler les statistiques
        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


# ==========================================
# 4. FONCTION DE VALIDATION
# ==========================================

def validate():

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

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


# ==========================================
# 5. BOUCLE PRINCIPALE
# ==========================================

def main():

    print("=" * 60)
    print("NEUROTRACE - RESNET18 TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Batch size:", train_loader.batch_size)

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch()

        val_loss, val_accuracy = validate()

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Sauvegarder le meilleur modèle
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": val_accuracy,
                    "class_to_idx": train_loader.dataset.class_to_idx,
                    "architecture": "ResNet18",
                },
                BEST_MODEL_PATH
            )

            print("Best model saved!")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Model saved at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
