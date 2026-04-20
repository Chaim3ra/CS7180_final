"""MLP encoder for static site metadata features."""

import torch
import torch.nn as nn

from src.models.base import BaseEncoder


class MetadataEncoder(BaseEncoder):
    """MLP encoder for static site-level features.

    Embeds time-invariant site characteristics — latitude, longitude, panel
    tilt, azimuth, system capacity, and elevation — into a ``d_model``-sized
    vector that is later concatenated with the fused temporal representation.

    The network uses a two-layer MLP with a hidden bottleneck:
    ``input_dim → hidden_dim → d_model``.

    Args:
        input_dim: Number of static features (default 6: lat, lon, tilt_deg,
            azimuth_deg, capacity_kw, elevation_m).
        hidden_dim: Hidden layer width.
        d_model: Output embedding dimension; should match the transformer
            encoders' ``d_model`` so the fusion module can concatenate cleanly.
        dropout: Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._output_dim = d_model

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    @property
    def output_dim(self) -> int:
        """Dimensionality of the metadata embedding (equals ``d_model``)."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed static site features.

        Args:
            x: Site feature matrix of shape ``(batch, input_dim)``.

        Returns:
            Site embedding of shape ``(batch, d_model)``.
        """
        return self.net(x)
