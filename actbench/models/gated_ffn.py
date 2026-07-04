"""SwiGLU-style gated FFN + a drop-in TransformerEncoderLayer using it.

SwiGLU replaces the two-linear FFN with a gated three-linear block:

    up_gate  = Linear(d_model, d_ff_gated)
    up_value = Linear(d_model, d_ff_gated)
    down     = Linear(d_ff_gated, d_model)

    y = down( SiLU(up_gate(x)) * up_value(x) )

Parameter and FLOPs match the standard FFN when
    d_ff_gated = 2/3 * d_ff_standard
(3 linears of half the hidden dim ≈ 2 linears of the original hidden dim).

We expose the SwiGLU output as a dedicated module so callbacks can hook it and
record post-activation moments the same way they do for pointwise activations.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden_dim(dim_feedforward: int) -> int:
    """Standard-parity SwiGLU hidden dim. Round to a multiple of 8 for GPU friendliness."""
    target = int(dim_feedforward * 2 / 3)
    rounded = int(math.ceil(target / 8) * 8)
    return max(rounded, 8)


class SwiGLU(nn.Module):
    """The gated feedforward block: SiLU(xW_gate) * (xW_value) -> W_down."""

    def __init__(self, d_model: int, d_ff_gated: int, dropout: float = 0.0):
        super().__init__()
        self.up_gate = nn.Linear(d_model, d_ff_gated, bias=False)
        self.up_value = nn.Linear(d_model, d_ff_gated, bias=False)
        self.down = nn.Linear(d_ff_gated, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.up_gate(x)) * self.up_value(x)
        return self.dropout(self.down(hidden))


class SwiGLUEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer whose FFN is a SwiGLU block.

    Mirrors nn.TransformerEncoderLayer(norm_first=True) so it can drop into
    existing model bodies (gpt_mini, vit_tiny, transformer_classifier).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = SwiGLU(d_model, swiglu_hidden_dim(dim_feedforward), dropout=dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Pre-norm attention with residual.
        normed = self.norm1(src)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        x = src + self.dropout1(attn_out)
        # Pre-norm SwiGLU FFN with residual.
        x = x + self.ffn(self.norm2(x))
        return x


class SwiGLUEncoderStack(nn.Module):
    """nn.TransformerEncoder-shaped wrapper around N SwiGLUEncoderLayers."""

    def __init__(self, layer: SwiGLUEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_clone_layer(layer) for _ in range(num_layers)])

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        return src


def _clone_layer(layer: SwiGLUEncoderLayer) -> SwiGLUEncoderLayer:
    d_model = layer.self_attn.embed_dim
    nhead = layer.self_attn.num_heads
    dropout = layer.dropout1.p
    d_ff_gated = layer.ffn.up_gate.out_features
    # Reverse-derive the "standard" dim_feedforward the layer was configured with.
    # (SwiGLUEncoderLayer stores d_ff_gated directly; hand it back through.)
    dim_feedforward = int(math.ceil(d_ff_gated * 3 / 2))
    return SwiGLUEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )
