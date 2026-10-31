
import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet18_Weights


# ============================================================
# 1. BUILD RESNET18
# ============================================================

def build_model(fine_tune=False):

    # Load pretrained ResNet18
    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the original FC layer (1000 classes)
    # with a new layer for our 4 MRI classes
    model.fc = nn.Linear(
        in_features=512,
        out_features=4
    )

    # The new classification layer is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    # Progressive fine-tuning
    if fine_tune:

        # Unfreeze Layer 4
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model


# ============================================================
# 2. DISPLAY MODEL INFORMATION
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(fine_tune=True)
    model = model.to(device)

    total_parameters = sum(
        param.numel()
        for param in model.parameters()
    )

    trainable_parameters = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    print("=" * 50)
    print("NEUROTRACE - RESNET18 FINE-TUNING")
    print("=" * 50)

    print("Device:", device)

    print("\nTrainable layers:")

    for name, param in model.named_parameters():

        if param.requires_grad:
            print(name)

    print("\nTotal parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    # Test forward pass
    dummy_input = torch.randn(
        2, 3, 224, 224
    ).to(device)

    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print("\nFine-tuning model ready!")
