"""Activation factory.

Adds two custom activations on top of the PyTorch built-ins:
- ScaledTanh — `alpha * tanh(x)`, tests whether tanh's underperformance is
  about bounded *range* (this should help) or bounded *shape* (this should not).
- Mish — `x * tanh(softplus(x))`, modern SiLU competitor.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_LEAKY_SLOPE = 0.01
_SCALED_TANH_ALPHA = 4.0


class ScaledTanh(nn.Module):
    def __init__(self, scale: float = _SCALED_TANH_ALPHA):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(x)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def get_activation(name: str) -> nn.Module:
    """Return a fresh activation module.

    `inplace=False` everywhere — inplace activations clash with
    `register_full_backward_hook` (used by GradientStatsCallback).

    `swiglu` is not a pointwise activation: it replaces the FFN block itself.
    Models must special-case activation_name == "swiglu" and swap in
    `actbench.models.gated_ffn.SwiGLUEncoderLayer` — never call this factory
    for that name.
    """
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky":
        return nn.LeakyReLU(_LEAKY_SLOPE)
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        return nn.GELU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key in ("scaled_tanh", "scaledtanh"):
        return ScaledTanh()
    if key == "mish":
        return Mish()
    if key == "swiglu":
        raise ValueError(
            "'swiglu' is a gated FFN, not a pointwise activation; construct a "
            "SwiGLUEncoderLayer from actbench.models.gated_ffn instead."
        )
    raise ValueError(
        f"Unsupported activation: {name}. "
        "Choose from tanh, relu, leaky, gelu, silu, scaled_tanh, mish, swiglu."
    )


def activation_names() -> list[str]:
    return ["tanh", "relu", "leaky", "gelu", "silu", "scaled_tanh", "mish", "swiglu"]
