"""Encoder-only transformer for AG News classification.

A small pre-norm encoder with a learned CLS-position embedding;
the CLS hidden state goes through a linear head.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .activations import get_activation
from .gated_ffn import SwiGLUEncoderLayer, SwiGLUEncoderStack


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        activation_name: str = "gelu",
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        if sequence_length > self.max_len:
            raise ValueError(f"Sequence length {sequence_length} exceeds max_len {self.max_len}.")

        positions = self.position_embedding[:, :sequence_length, :]
        hidden = self.token_embedding(token_ids) * math.sqrt(self.d_model) + positions
        hidden = self.embedding_dropout(hidden)
        hidden = self.blocks(hidden)
        hidden = self.final_norm(hidden)
        # Mean-pool over the sequence — robust for fixed-length padded inputs.
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)
