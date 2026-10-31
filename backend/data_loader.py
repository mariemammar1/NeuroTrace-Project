
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "splits"

BATCH_SIZE = 16
IMAGE_SIZE = 224

# Utiliser le GPU si disponible
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# 2. TRANSFORMATIONS DES IMAGES
# ==========================================

image_transforms = transforms.Compose([

    # Redimensionner toutes les images
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Convertir les images en RGB (3 canaux)
    transforms.Lambda(lambda image: image.convert("RGB")),

    # Convertir l'image en tenseur PyTorch
    transforms.ToTensor(),

    # Normalisation selon ImageNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. CHARGEMENT DES DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    root=DATASET_DIR / "train",
    transform=image_transforms
)

validation_dataset = datasets.ImageFolder(
    root=DATASET_DIR / "validation",
    transform=image_transforms
)

test_dataset = datasets.ImageFolder(
    root=DATASET_DIR / "test",
    transform=image_transforms
)


# ==========================================
# 4. VERIFICATION DES CLASSES
# ==========================================

print("\nClass mapping:")
print(train_dataset.class_to_idx)

# Vérifier que les trois ensembles utilisent
# exactement le même ordre de classes
assert train_dataset.class_to_idx == validation_dataset.class_to_idx
assert train_dataset.class_to_idx == test_dataset.class_to_idx

print("\nNumber of images:")
print("Train:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))


# ==========================================
# 5. CREATION DES DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ==========================================
# 6. TEST DU CHARGEMENT
# ==========================================
if __name__ == "__main__":
  images, labels = next(iter(train_loader))

  print("\nFirst batch:")
  print("Images shape:", images.shape)
  print("Labels shape:", labels.shape)
  print("Labels:", labels)

  # Envoyer un batch sur le GPU pour vérifier
  images = images.to(DEVICE)
  labels = labels.to(DEVICE)

  print("\nGPU verification:")
  print("Images device:", images.device)
  print("Labels device:", labels.device)

  print("\nDataLoader test completed successfully!")
