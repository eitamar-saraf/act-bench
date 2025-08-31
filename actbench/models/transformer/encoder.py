from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from .backbone import TransformerBackbone

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, num_classes: int = 4, dim_feedforward: int = 256, activation_name: str = "relu", max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = TransformerBackbone(vocab_size, d_model, nhead, num_layers, dim_feedforward, activation_name, max_len, dropout, is_decoder=False)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.backbone(src, src_key_padding_mask=src_key_padding_mask)
        cls = out[:, 0, :]
        return self.classifier(cls)
