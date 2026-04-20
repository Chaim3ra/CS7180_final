"""MLP regression head for multi-step solar generation forecasting."""

import torch
import torch.nn as nn

from src.models.base import BaseHead


class RegressionHead(BaseHead):
    """Multi-layer perceptron that maps a fused representation to kWh forecasts.

    Produces one predicted kWh value per 15-minute interval in the forecast
    horizon via a two-layer MLP:
    ``input_dim → hidden_dim → forecast_horizon``.

    Args:
        input_dim: Dimensionality of the fused input vector (typically
            ``2 * d_model`` from :class:`~src.models.fusion.CrossAttentionFusion`).
        hidden_dim: Width of the single hidden layer.
        forecast_horizon: Number of 15-min intervals to predict.
        dropout: Dropout probability applied before the output layer.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        forecast_horizon: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future solar generation values.

        Args:
            x: Fused representation of shape ``(batch, input_dim)``.

        Returns:
            Forecast tensor of shape ``(batch, forecast_horizon)`` where each
            value represents predicted kWh for one 15-min interval.
        """
        return self.net(x)
