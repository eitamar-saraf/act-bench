"""Weight initialization, selected by architecture family."""
from __future__ import annotations
import torch.nn as nn


def initialize_weights(model: nn.Module, scheme: str = "xavier") -> None:
    """Apply weight init in-place.

    Schemes:
    - "xavier" : transformers / linear+embedding heavy nets.
    - "kaiming": CNNs (Kaiming-normal, fan_in, relu nonlinearity).
    """
    if scheme == "xavier":
        model.apply(_xavier_init)
    elif scheme == "kaiming":
        model.apply(_kaiming_init)
    else:
        raise ValueError(f"Unknown init scheme: {scheme}")


def _xavier_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)


def _kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
