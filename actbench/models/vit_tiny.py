"""ViT-tiny for CIFAR-10. Activation pluggable in the FFN (same pattern as gpt_mini)."""
from __future__ import annotations
import torch
import torch.nn as nn

from .activations import get_activation
from .gated_ffn import SwiGLUEncoderLayer, SwiGLUEncoderStack


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, d_model, H/P, W/P) -> (B, num_patches, d_model)
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class ViTTiny(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        d_model: int = 192,
        nhead: int = 3,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        activation_name: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.embedding_dropout = nn.Dropout(dropout)

        if activation_name.lower() == "swiglu":
            self.blocks = SwiGLUEncoderStack(
                SwiGLUEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
                num_layers=num_layers,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=get_activation(activation_name),
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls_tokens, patches], dim=1) + self.position_embedding
        hidden = self.embedding_dropout(hidden)
        hidden = self.blocks(hidden)
        hidden = self.final_norm(hidden)
        return self.classifier(hidden[:, 0])
