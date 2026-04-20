"""Cross-attention fusion between weather and generation encoders."""

import torch
import torch.nn as nn

from src.models.base import BaseFusion


class CrossAttentionFusion(BaseFusion):
    """Fuse weather, generation, and metadata representations.

    Cross-attention is applied with the generation sequence as *query* and the
    weather sequence as *key* and *value*, letting the model learn which weather
    time steps are most relevant to each generation step.  The attended output
    is mean-pooled to a single vector, then concatenated with the static
    metadata embedding to form the final fused representation:

    .. code-block:: text

        weather_enc  (B, T, d_model)  ─────────────────────┐
                                                            │  K, V
        gen_enc      (B, T, d_model)  ── Q ── CrossAttn ── pool ─┐
                                                                  │ concat → (B, 2*d_model)
        metadata_enc (B, d_model)     ───────────────────────────┘

    Args:
        d_model: Dimension shared by all three encoder outputs.
        nhead: Number of attention heads for cross-attention (must divide
            ``d_model`` evenly).
        dropout: Dropout probability inside the attention layer.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self._output_dim = d_model * 2

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    @property
    def output_dim(self) -> int:
        """Output dimension: ``2 * d_model`` (attended + metadata)."""
        return self._output_dim

    def forward(
        self,
        weather_enc: torch.Tensor,
        gen_enc: torch.Tensor,
        metadata_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse the three modality representations.

        Args:
            weather_enc: Weather encoder output of shape
                ``(batch, seq_len, d_model)``.
            gen_enc: Generation encoder output of shape
                ``(batch, seq_len, d_model)``.
            metadata_enc: Metadata encoder output of shape
                ``(batch, d_model)``.

        Returns:
            Fused tensor of shape ``(batch, 2 * d_model)``.
        """
        # generation attends to weather
        attended, _ = self.cross_attn(
            query=gen_enc,       # (B, T, d_model)
            key=weather_enc,     # (B, T, d_model)
            value=weather_enc,   # (B, T, d_model)
        )
        attended = self.norm(attended + gen_enc)  # residual + norm

        pooled = attended.mean(dim=1)             # (B, d_model)
        fused = torch.cat([pooled, metadata_enc], dim=-1)  # (B, 2*d_model)
        return fused
