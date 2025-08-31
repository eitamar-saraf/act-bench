"""actbench package: activation function benchmarking framework.

Exports key factory functions for convenience.
"""
from .models.activations import get_activation, activation_names  # noqa: F401
__all__ = ["get_activation", "activation_names"]
