"""Minimal ResNet-18 with pluggable activation.

We re-implement a lightweight variant to ensure every ReLU is replaceable.
Only basic building blocks needed for Tiny-ImageNet (num_classes=200 by default).
"""
from __future__ import annotations
from typing import Type, Callable, Optional
import torch
import torch.nn as nn
from models.activations import get_activation

# Kaiming fan_in initialization applied manually so it is consistent across activations.

def kaiming_fanin_(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, activation: nn.Module, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = activation  # reused instance OK (stateless) or pass new each time; we clone by building per block.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock], layers: list[int], num_classes: int = 200, activation_name: str = "relu"):
        super().__init__()
        self.in_planes = 64
        self.activation_name = activation_name
        self.act_factory = lambda: get_activation(activation_name)  # create fresh module when called

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = self.act_factory()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights consistently.
        self.apply(kaiming_fanin_)
        # Explicit BN init
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, self.act_factory(), stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, self.act_factory()))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(activation: str = "relu", num_classes: int = 200) -> ResNet:
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, activation_name=activation)

if __name__ == "__main__":
    model = resnet18("gelu", num_classes=10)
    x = torch.randn(2,3,64,64)
    y = model(x)
    print(y.shape)
