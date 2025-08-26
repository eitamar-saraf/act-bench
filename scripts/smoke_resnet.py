"""Quick smoke test for pluggable ResNet-18 activations."""
from __future__ import annotations
import torch
from models.resnet import resnet18
from models.activations import activation_names


def main():
    x = torch.randn(2,3,64,64)
    for name in activation_names():
        model = resnet18(name, num_classes=10)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2,10), f"Bad output shape for {name}: {y.shape}"
        print(f"{name}: OK")

if __name__ == "__main__":
    main()
