"""CIFAR-scale ResNet-20 with pluggable activation.

Follows He et al. 2016: 3 stages × 3 BasicBlocks, channels [16, 32, 64],
strided downsample at the start of stages 2 and 3. ~270K params.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .activations import get_activation


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, activation_name: str):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = get_activation(activation_name)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = get_activation(activation_name)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.act2(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes: int = 10, activation_name: str = "relu"):
        if activation_name.lower() == "swiglu":
            raise ValueError(
                "SwiGLU is a gated FFN; ResNet-20 has no FFN block. "
                "Restrict the grid to transformer tasks (lm, cls, lm_large, vision_vit) for swiglu."
            )
        super().__init__()
        self.stem_conv = _conv3x3(3, 16)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_act = get_activation(activation_name)

        self.stage1 = self._make_stage(16, 16, blocks=3, stride=1, activation_name=activation_name)
        self.stage2 = self._make_stage(16, 32, blocks=3, stride=2, activation_name=activation_name)
        self.stage3 = self._make_stage(32, 64, blocks=3, stride=2, activation_name=activation_name)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def _make_stage(self, in_channels: int, out_channels: int, blocks: int, stride: int, activation_name: str) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride, activation_name)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, activation_name=activation_name))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
