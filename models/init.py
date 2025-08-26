"""Centralized weight initialization utilities for consistent model setup.

Provides policies for initializing different model types (CNNs, Transformers)
as specified in the project checklist.
"""
from __future__ import annotations
import torch.nn as nn

def kaiming_fanin_init(module: nn.Module):
    """Kaiming Normal (fan-in) for Conv2D and Linear layers."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def xavier_uniform_init(module: nn.Module):
    """Xavier Uniform for Linear layers, common for Transformers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def apply_init(model: nn.Module, policy: str):
    """
    Apply a named initialization policy to a model.

    Args:
        model: The model to initialize.
        policy: The name of the policy ('kaiming_fanin' or 'xavier_uniform').
    """
    if policy == "kaiming_fanin":
        model.apply(kaiming_fanin_init)
    elif policy == "xavier_uniform":
        model.apply(xavier_uniform_init)
    else:
        raise ValueError(f"Unknown initialization policy: {policy}")

if __name__ == "__main__":
    # Smoke test
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.linear1 = nn.Linear(16, 10)

    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 5)

    cnn = SimpleCNN()
    apply_init(cnn, "kaiming_fanin")
    print("Kaiming fan-in init applied to CNN.")

    transformer = SimpleTransformer()
    apply_init(transformer, "xavier_uniform")
    print("Xavier uniform init applied to Transformer.")
