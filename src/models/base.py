"""Abstract base classes for all model components."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all input encoders.

    Subclasses must implement :meth:`forward` and expose :attr:`output_dim`
    so downstream fusion modules can query the encoder's output size without
    instantiating a dummy tensor.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input tensor.

        Args:
            x: Input tensor. Shape conventions are encoder-specific but are
               documented in each concrete subclass.

        Returns:
            Encoded representation tensor.
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the last axis of this encoder's output tensor."""


class BaseFusion(nn.Module, ABC):
    """Abstract base class for fusion modules.

    A fusion module combines representations from multiple encoders into a
    single vector that is passed to a prediction head.
    """

    @abstractmethod
    def forward(self, *encodings: torch.Tensor) -> torch.Tensor:
        """Fuse one or more encoder outputs into a single representation.

        Args:
            *encodings: Encoder output tensors. Ordering and shapes are
                        defined by each concrete subclass.

        Returns:
            Fused representation tensor of shape ``(batch, output_dim)``.
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the fused output vector."""


class BaseHead(nn.Module, ABC):
    """Abstract base class for prediction heads.

    A head maps a fused representation to the final model output (e.g. a
    sequence of forecasted values).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a fused representation to predictions.

        Args:
            x: Fused tensor of shape ``(batch, input_dim)``.

        Returns:
            Prediction tensor. Shape is head-specific.
        """
