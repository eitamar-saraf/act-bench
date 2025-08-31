from __future__ import annotations
import math
import torch
import torch.nn as nn
from ..activations import get_activation

class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, activation_name: str, max_len: int, dropout: float, is_decoder: bool = False):
        super().__init__()
        self.d_model = d_model
        self.is_decoder = is_decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        act_mod = get_activation(activation_name)
        if is_decoder:
            layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=act_mod, batch_first=True)
            self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        else:
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=act_mod, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        b, s = src.shape
        pos = self.pos_embed[:, :s, :]
        x = self.embedding(src) * math.sqrt(self.d_model) + pos
        if self.is_decoder:
            if src_mask is None:
                src_mask = nn.Transformer.generate_square_subsequent_mask(s, device=src.device)
            return self.transformer(tgt=x, memory=x, tgt_mask=src_mask, tgt_key_padding_mask=src_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        else:
            return self.transformer(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
