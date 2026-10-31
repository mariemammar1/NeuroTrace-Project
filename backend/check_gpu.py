
import torch
import torchvision

print("=" * 50)
print("NEUROTRACE - PYTORCH GPU CHECK")
print("=" * 50)

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")

    # Petit calcul de test sur le GPU
    a = torch.rand(1000, 1000, device="cuda")
    b = torch.rand(1000, 1000, device="cuda")
    result = torch.matmul(a, b)

    print("GPU computation: SUCCESS")
    print("Result device:", result.device)
else:
    print("CUDA is not available. Check installation.")

print("=" * 50)
