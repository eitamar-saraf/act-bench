"""Minimal Transformer Encoder for AG News classification using a shared backbone."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from models.backbone import TransformerBackbone

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, num_classes: int = 4, dim_feedforward: int = 256, activation_name: str = "relu", max_len: int = 128, dropout: float = 0.1):
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
            is_decoder=False
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.backbone(src, src_key_padding_mask=src_key_padding_mask)
        # Use the output of the [CLS] token (first token) for classification
        cls_output = output[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

if __name__ == "__main__":
    # Smoke test
    vocab_size = 10000
    batch_size = 2
    seq_len = 32
    num_classes = 4
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Create a padding mask (True where padded)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, 20:] = True # Example: first sample is padded from index 20 onwards

    model = TransformerEncoderClassifier(
        vocab_size=vocab_size, 
        num_classes=num_classes, 
        activation_name="gelu", 
        max_len=64
    )
    
    logits = model(x, src_key_padding_mask=mask)
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, num_classes)
    print("Transformer with shared backbone OK")
