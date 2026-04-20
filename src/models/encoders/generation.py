"""Transformer encoder for historical solar generation time-series."""

import math

import torch
import torch.nn as nn

from src.models.base import BaseEncoder


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Model dimension.
        dropout: Dropout probability.
        max_len: Maximum sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of shape ``(batch, seq_len, d_model)``.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GenerationEncoder(BaseEncoder):
    """Transformer encoder for past solar generation history.

    Encodes a univariate sequence of historical kWh readings (one value per
    15-min interval) into a per-step representation suitable for cross-attention
    with weather features.

    Args:
        input_dim: Number of generation features per step (default 1 for kWh).
        d_model: Internal transformer hidden dimension.
        nhead: Number of multi-head attention heads.
        num_layers: Number of :class:`~torch.nn.TransformerEncoderLayer` blocks.
        ffn_dim: Feed-forward network hidden dimension inside each layer.
        dropout: Dropout probability used throughout.
        max_len: Maximum supported sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self._output_dim = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoded output (equals ``d_model``)."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a generation history sequence.

        Args:
            x: Past generation values of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Encoded sequence of shape ``(batch, seq_len, d_model)``.
        """
        x = self.input_proj(x)   # (B, T, d_model)
        x = self.pos_enc(x)      # (B, T, d_model)
        x = self.transformer(x)  # (B, T, d_model)
        return x
