
import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet18_Weights


# ==========================================
# 1. CONFIGURATION
# ==========================================

NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ==========================================
# 2. CONSTRUCTION DE RESNET18
# ==========================================

def build_model(num_classes=NUM_CLASSES):

    # Charger ResNet18 avec les poids pré-entraînés
    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Geler toutes les couches du réseau
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Remplacer la dernière couche
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes
    )

    # La nouvelle couche doit être entraînable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


# ==========================================
# 3. TEST DU MODELE
# ==========================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 50)
    print("NEUROTRACE - RESNET18 MODEL")
    print("=" * 50)

    print("Device:", device)

    # Construire le modèle
    model = build_model()

    # Envoyer le modèle sur le GPU
    model = model.to(device)

    print("\nModel loaded successfully!")

    # Afficher la dernière couche
    print("\nFinal layer:")
    print(model.fc)

    # Créer un batch fictif de 16 images
    sample_images = torch.randn(
        16, 3, 224, 224
    ).to(device)

    # Mode évaluation pour ce test
    model.eval()

    with torch.no_grad():

        outputs = model(sample_images)

    print("\nForward propagation test:")
    print("Input shape:", sample_images.shape)
    print("Output shape:", outputs.shape)

    # Convertir les scores en probabilités
    probabilities = torch.softmax(outputs, dim=1)

    print("\nProbabilities for first image:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name}: "
            f"{probabilities[0][index].item() * 100:.2f}%"
        )

    # Vérifier que les probabilités totalisent 1
    print(
        "\nProbability sum:",
        probabilities[0].sum().item()
    )

    print("\nTrainable parameters:")

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Trainable: {trainable:,}")
    print(f"Total: {total:,}")

    print("\nResNet18 test completed successfully!")
