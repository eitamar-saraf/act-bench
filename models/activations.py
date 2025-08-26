"""Activation factory for the activation comparison study.

Supported keys: tanh, relu, leaky, gelu, silu
Leaky slope fixed at 0.01.
"""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn

_LEAKY_SLOPE = 0.01


def get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "leaky":
        return nn.LeakyReLU(_LEAKY_SLOPE, inplace=True)
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        # Use PyTorch GELU (approximate = False for exact tanh based? default = False uses exact erf impl in recent versions)
        return nn.GELU()
    if key in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}. Choose from tanh,relu,leaky,gelu,silu.")


def activation_names() -> list[str]:
    return ["tanh", "relu", "leaky", "gelu", "silu"]


if __name__ == "__main__":  # simple smoke test
    for n in activation_names():
        act = get_activation(n)
        x = torch.randn(4, 5)
        y = act(x)
        assert y.shape == x.shape
    print("Activation factory OK")
