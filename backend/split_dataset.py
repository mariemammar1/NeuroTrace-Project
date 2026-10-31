
from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split
import csv

# Racine du projet NeuroTrace
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DIR = PROJECT_ROOT / "dataset" / "cleaned"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# 80 % pour l'entraînement, 20 % pour la validation
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

def main():
    if not CLEANED_DIR.exists():
        raise FileNotFoundError(
            f"Dataset nettoyé introuvable : {CLEANED_DIR}"
        )

    if SPLITS_DIR.exists():
        raise FileExistsError(
            f"{SPLITS_DIR} existe déjà. "
            "Renomme ou supprime ce dossier avant de relancer."
        )

    print("=" * 55)
    print("NEUROTRACE - STRATIFIED DATASET SPLIT")
    print("=" * 55)

    report = []

    for class_name in CLASSES:
        source_train = CLEANED_DIR / "Training" / class_name
        source_test = CLEANED_DIR / "Testing" / class_name

        if not source_train.exists() or not source_test.exists():
            raise FileNotFoundError(
                f"Dossier manquant pour la classe {class_name}"
            )

        # Récupérer les images nettoyées de Training
        images = sorted(
            p for p in source_train.iterdir()
            if p.is_file()
        )

        # Séparation stratifiée : une seule classe à la fois
        # Les proportions par classe sont donc conservées.
        train_images, val_images = train_test_split(
            images,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # Créer les dossiers de destination
        destinations = {
            "train": train_images,
            "validation": val_images,
            "test": sorted(
                p for p in source_test.iterdir()
                if p.is_file()
            )
        }

        for split_name, files in destinations.items():
            destination = SPLITS_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in files:
                copy2(image_path, destination / image_path.name)

            report.append({
                "split": split_name,
                "class": class_name,
                "images": len(files)
            })

            print(
                f"{split_name}/{class_name}: "
                f"{len(files)} images"
            )

    # Sauvegarder un rapport CSV
    report_path = PROJECT_ROOT / "docs" / "split_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "class", "images"]
        )
        writer.writeheader()
        writer.writerows(report)

    print("\n" + "=" * 55)
    print("SPLIT COMPLETED")
    print("=" * 55)

    for split_name in ["train", "validation", "test"]:
        total = sum(
            row["images"]
            for row in report
            if row["split"] == split_name
        )
        print(f"{split_name}: {total} images")

    print(f"\nReport saved to: {report_path}")
    print(f"Dataset splits: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
