
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from model import build_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT / "models" / "resnet18_finetuned_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "docs" / "gradcam"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Same preprocessing used during model training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 2. LOAD THE FINE-TUNED MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = build_model(fine_tune=True)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print("Fine-tuned model loaded.")
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Validation accuracy:",
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    return model


# ============================================================
# 3. GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register a forward hook on the target layer
        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, inputs, output):

        self.activations = output

        # Save gradients of the target layer output
        output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):

        self.gradients = gradient

    def generate(self, input_tensor, class_index=None):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        # Forward pass
        output = self.model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        # Use the predicted class unless another class is requested
        if class_index is None:
            class_index = predicted_index

        # Score of the selected class before softmax
        class_score = output[:, class_index]

        # Backward pass
        class_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Could not capture Grad-CAM activations or gradients."
            )

        # Average gradients over spatial dimensions
        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of activation maps
        cam = torch.sum(
            weights * self.activations,
            dim=1
        )

        # Keep positive contributions
        cam = F.relu(cam)

        # Resize heatmap to model input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        # Normalize between 0 and 1
        cam_min = cam.min()
        cam_max = cam.max()

        cam = (cam - cam_min) / (
            cam_max - cam_min + 1e-8
        )

        return (
            cam.detach().cpu().numpy(),
            predicted_index,
            probabilities.detach().cpu().squeeze().numpy()
        )

    def remove_hooks(self):

        self.forward_handle.remove()


# ============================================================
# 4. LOAD AND PREPROCESS MRI IMAGE
# ============================================================

def load_mri(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"MRI image not found: {image_path}"
        )

    image = Image.open(image_path)

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Keep an original-size copy for visualization
    original_image = image.copy()

    # Apply exactly the same preprocessing as training
    input_tensor = preprocess(image).unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    return original_image, input_tensor


# ============================================================
# 5. GENERATE AND SAVE VISUALIZATIONS
# ============================================================

def save_visualizations(
    original_image,
    heatmap,
    predicted_class,
    confidence,
    image_path
):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    # Convert heatmap to an RGB image
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    # Save original image
    original_path = (
        OUTPUT_DIR / f"{image_name}_original.png"
    )

    original_image.save(original_path)

    # Save heatmap separately
    heatmap_path = (
        OUTPUT_DIR / f"{image_name}_heatmap.png"
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        heatmap_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    # Resize heatmap to original MRI dimensions
    heatmap_resized = np.array(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # Create overlay
    overlay_path = (
        OUTPUT_DIR / f"{image_name}_overlay.png"
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(original_image)

    plt.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1
    )

    plt.title(
        f"Prediction: {predicted_class} "
        f"({confidence * 100:.2f}%)"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        overlay_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nVisualizations saved:")
    print("Original:", original_path)
    print("Heatmap:", heatmap_path)
    print("Overlay:", overlay_path)


# ============================================================
# 6. MAIN
# ============================================================

def main(image_path):

    print("=" * 60)
    print("NEUROTRACE - GRAD-CAM EXPLAINABILITY")
    print("=" * 60)

    print("Device:", DEVICE)

    model = load_model()

    original_image, input_tensor = load_mri(image_path)

    # ResNet18's last convolutional block
    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    heatmap, predicted_index, probabilities = gradcam.generate(
        input_tensor
    )

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index]

    print("\nPrediction:", predicted_class)
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nClass probabilities:")

    for index, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name:12}: "
            f"{probabilities[index] * 100:.2f}%"
        )

    save_visualizations(
        original_image=original_image,
        heatmap=heatmap,
        predicted_class=predicted_class,
        confidence=confidence,
        image_path=image_path
    )

    gradcam.remove_hooks()

    print("\nGRAD-CAM COMPLETED")


# ============================================================
# 7. COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for a brain MRI image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the MRI image."
    )

    args = parser.parse_args()

    main(args.image)
