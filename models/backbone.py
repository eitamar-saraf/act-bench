"""A shared transformer backbone for encoder and decoder models."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from models.activations import get_activation
from models.layers import PositionalEncoding
from models.init import apply_init

class TransformerBackbone(nn.Module):
    """A shared transformer backbone that can be configured as an encoder or a decoder."""
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, activation_name: str, max_len: int, dropout: float, is_decoder: bool = False):
        super().__init__()
        self.d_model = d_model
        self.is_decoder = is_decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        activation_module = get_activation(activation_name)
        
        if is_decoder:
            layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation_module, batch_first=True
            )
            self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation_module, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            
        apply_init(self, "xavier_uniform")

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        
        if self.is_decoder:
            if src_mask is None:
                size = src.size(1)
                src_mask = nn.Transformer.generate_square_subsequent_mask(size, device=src.device)
            
            return self.transformer(
                tgt=src, memory=src, tgt_mask=src_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
        else:
            return self.transformer(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
