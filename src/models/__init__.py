"""Multi-modal solar generation forecasting model package.

Component registry and ``build()`` factory.  Usage::

    from src.models import build
    model = build("configs/experiment.yaml")
    preds = model(weather, generation, metadata)
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import yaml

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]

from src.models.encoders import GenerationEncoder, MetadataEncoder, WeatherEncoder
from src.models.fusion import CrossAttentionFusion
from src.models.heads import RegressionHead

__all__ = ["SolarForecastModel", "build"]


class SolarForecastModel(L.LightningModule):
    """Full multi-modal solar generation forecasting model.

    Combines three modality-specific encoders, a cross-attention fusion module,
    and an MLP regression head.  Subclasses
    :class:`lightning.LightningModule` to provide training, validation, test,
    and optimiser logic for use with the Lightning :class:`~lightning.Trainer`.

    Data flow::

        weather    (B, T, W) ──► WeatherEncoder    ──┐
        generation (B, T, 1) ──► GenerationEncoder ──┼──► CrossAttentionFusion ──► RegressionHead ──► (B, H)
        metadata   (B, M)    ──► MetadataEncoder   ──┘

    where ``T`` = seq_len, ``W`` = weather_input_dim, ``M`` = metadata_input_dim,
    and ``H`` = forecast_horizon.

    Args:
        weather_encoder: Encoder for meteorological time-series.
        generation_encoder: Encoder for historical generation time-series.
        metadata_encoder: Encoder for static site features.
        fusion: Cross-attention fusion module.
        head: Regression prediction head.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        scheduler: LR scheduler type — ``"cosine"``, ``"step"``, or ``"none"``.
        epochs: Total training epochs (used by cosine annealer as ``T_max``).
    """

    def __init__(
        self,
        weather_encoder: WeatherEncoder,
        generation_encoder: GenerationEncoder,
        metadata_encoder: MetadataEncoder,
        fusion: CrossAttentionFusion,
        head: RegressionHead,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler: str = "cosine",
        epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["weather_encoder", "generation_encoder",
                    "metadata_encoder", "fusion", "head"]
        )
        self.weather_encoder    = weather_encoder
        self.generation_encoder = generation_encoder
        self.metadata_encoder   = metadata_encoder
        self.fusion             = fusion
        self.head               = head

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        weather: torch.Tensor,
        generation: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            weather: Meteorological time-series of shape
                ``(batch, seq_len, weather_input_dim)``.
            generation: Historical generation time-series of shape
                ``(batch, seq_len, generation_input_dim)``.
            metadata: Static site features of shape
                ``(batch, metadata_input_dim)``.

        Returns:
            Predicted kWh values of shape ``(batch, forecast_horizon)``.
        """
        weather_enc = self.weather_encoder(weather)         # (B, T, d_model)
        gen_enc     = self.generation_encoder(generation)   # (B, T, d_model)
        meta_enc    = self.metadata_encoder(metadata)       # (B, d_model)
        fused       = self.fusion(weather_enc, gen_enc, meta_enc)  # (B, 2*d_model)
        return self.head(fused)                             # (B, forecast_horizon)

    # ------------------------------------------------------------------
    # Lightning step hooks
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        weather, generation, metadata, targets = batch
        preds = self(weather, generation, metadata)
        loss  = F.mse_loss(preds, targets)
        mae   = F.l1_loss(preds, targets)
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae",  mae,  on_step=False,
                 on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Compute MSE loss for one training batch.

        Args:
            batch: Tuple of ``(weather, generation, metadata, targets)``.
            batch_idx: Index of the current batch.

        Returns:
            Scalar MSE loss tensor.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        """Log validation MSE and MAE without returning a loss.

        Args:
            batch: Tuple of ``(weather, generation, metadata, targets)``.
            batch_idx: Index of the current batch.
        """
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        """Log test MSE and MAE.

        Args:
            batch: Tuple of ``(weather, generation, metadata, targets)``.
            batch_idx: Index of the current batch.
        """
        self._shared_step(batch, "test")

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Build AdamW optimiser and optional LR scheduler.

        Returns:
            Either a bare optimiser dict (``"none"`` scheduler) or a dict
            with ``"optimizer"`` and ``"lr_scheduler"`` keys for Lightning.
        """
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.epochs
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }
        if self.hparams.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }
        return opt


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build(config_path: Union[str, Path]) -> SolarForecastModel:
    """Construct a :class:`SolarForecastModel` from a YAML config file.

    Reads ``configs/experiment.yaml`` (or any compatible path) and
    instantiates each sub-module with the specified hyperparameters.

    Args:
        config_path: Path to the experiment YAML file.

    Returns:
        Fully constructed :class:`SolarForecastModel` (not yet trained).

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        KeyError: If required config keys are missing.

    Example::

        model = build("configs/experiment.yaml")
        print(model)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]
    t = cfg["training"]
    d_model    = m["d_model"]
    nhead      = m["nhead"]
    num_layers = m["num_layers"]
    dropout    = m["dropout"]

    weather_encoder = WeatherEncoder(
        input_dim=m["encoders"]["weather"]["input_dim"],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ffn_dim=m["encoders"]["weather"]["ffn_dim"],
        dropout=dropout,
    )
    generation_encoder = GenerationEncoder(
        input_dim=m["encoders"]["generation"]["input_dim"],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ffn_dim=m["encoders"]["generation"]["ffn_dim"],
        dropout=dropout,
    )
    metadata_encoder = MetadataEncoder(
        input_dim=m["encoders"]["metadata"]["input_dim"],
        hidden_dim=m["encoders"]["metadata"]["hidden_dim"],
        d_model=d_model,
        dropout=dropout,
    )
    fusion = CrossAttentionFusion(
        d_model=d_model,
        nhead=m["fusion"]["nhead"],
        dropout=dropout,
    )
    head = RegressionHead(
        input_dim=fusion.output_dim,
        hidden_dim=m["head"]["hidden_dim"],
        forecast_horizon=m["head"]["forecast_horizon"],
        dropout=dropout,
    )

    return SolarForecastModel(
        weather_encoder=weather_encoder,
        generation_encoder=generation_encoder,
        metadata_encoder=metadata_encoder,
        fusion=fusion,
        head=head,
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        scheduler=t["scheduler"],
        epochs=t["epochs"],
    )
