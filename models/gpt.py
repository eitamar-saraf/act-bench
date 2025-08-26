"""Minimal GPT-style Decoder for Language Modeling using a shared backbone."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from models.backbone import TransformerBackbone

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 512, activation_name: str = "relu", max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = TransformerBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            activation_name=activation_name,
            max_len=max_len,
            dropout=dropout,
            is_decoder=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.backbone(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(output)
        return logits

if __name__ == "__main__":
    # Smoke test
    vocab_size = 10000
    batch_size = 4
    seq_len = 64
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    model = GPTDecoder(
        vocab_size=vocab_size,
        activation_name="gelu",
        max_len=128
    )
    
    logits = model(x)
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("GPT-mini Decoder with shared backbone OK")
