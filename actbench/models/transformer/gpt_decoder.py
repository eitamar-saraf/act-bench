from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from .backbone import TransformerBackbone

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 512, activation_name: str = "relu", max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = TransformerBackbone(vocab_size, d_model, nhead, num_layers, dim_feedforward, activation_name, max_len, dropout, is_decoder=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.backbone(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(out)
