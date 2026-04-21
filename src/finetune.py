"""src/finetune.py — NY transfer fine-tuning for the solar forecasting model.

Loads a pre-trained checkpoint (V1 or V2), evaluates zero-shot on the full
New York held-out dataset, fine-tunes with a small NY sample using frozen
weather/generation encoders, then reports how much of the generalisation gap
is closed.

Frozen modules  : weather_encoder, generation_encoder
Trainable       : metadata_encoder, fusion, head
Rationale       : Weather-to-generation physics transfer should already be
                  learned; only the site-adaptation layers need NY exposure.

Usage
-----
    python src/finetune.py --ny_days 7  --model_version v1
    python src/finetune.py --ny_days 30 --model_version v1
    python src/finetune.py --ny_days 90 --checkpoint s3://... --model_version v2

S3 credentials required (set in .env):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=us-east-2
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present
_env = ROOT / ".env"
if _env.exists():
    for _raw in _env.read_text().splitlines():
        _raw = _raw.strip()
        if _raw and not _raw.startswith("#") and "=" in _raw:
            _k, _v = _raw.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore[no-redef]
    from pytorch_lightning.loggers import CSVLogger  # type: ignore[no-redef]

from src.models import SolarForecastModel, build
from src.dataloader import SolarWindowDataset, read_parquet, ensure_local_parquet
from src.evaluate import evaluate_from_loader, evaluate_parquet
from src.results_utils import auto_commit, save_per_home, save_row

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEATHER_COLS  = ["GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct"]
SOLAR_COL     = "solar_kwh"
METADATA_COLS = ["lat", "lon", "tilt_deg", "azimuth_deg", "capacity_kw", "elevation_m"]
STEPS_PER_DAY = 96   # 15-min resolution → 96 steps per 24 h

DEFAULT_CHECKPOINT = (
    "s3://cs7180-final-project/checkpoints/"
    "2026-04-21_01-40-26/solar-epoch=04-val_loss=0.0042.ckpt"
)

FT_LR = 1e-4


# ---------------------------------------------------------------------------
# Shared callbacks (mirror train.py)
# ---------------------------------------------------------------------------

class EpochSummaryCallback(L.Callback):
    """Prints one clean line per epoch; designed for enable_progress_bar=False."""

    def __init__(self, max_epochs: int) -> None:
        self._max_epochs = max_epochs
        self._t0: float  = 0.0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        m     = trainer.callback_metrics
        epoch = trainer.current_epoch + 1
        print(
            f"Epoch {epoch:>3}/{self._max_epochs}"
            f" | train_loss: {float(m.get('train_loss', float('nan'))):.4f}"
            f" | val_loss: {float(m.get('val_loss', float('nan'))):.4f}"
            f" | MAE: {float(m.get('val_mae', float('nan'))):.4f}"
            f" | {int(time.time() - self._t0)}s",
            flush=True,
        )


class VerboseEarlyStopping(EarlyStopping):
    """EarlyStopping that prints a message when it fires."""

    def on_validation_end(self, trainer, pl_module) -> None:
        super().on_validation_end(trainer, pl_module)
        if trainer.should_stop:
            print(f"\n  Early stopping triggered at epoch {trainer.current_epoch}", flush=True)


class S3ModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that mirrors every saved file to S3 in real-time."""

    def __init__(self, *args, s3_bucket: str = "", s3_prefix: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix.rstrip("/")

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if self._s3_bucket and Path(filepath).exists():
            self._upload(filepath)

    def _upload(self, filepath: str) -> None:
        import boto3
        filename = Path(filepath).name
        s3_key   = f"{self._s3_prefix}/{filename}"
        boto3.client("s3").upload_file(filepath, self._s3_bucket, s3_key)
        print(f"  Checkpoint → s3://{self._s3_bucket}/{s3_key}", flush=True)


# ---------------------------------------------------------------------------
# Fine-tuning model — frozen encoders, trainable head
# ---------------------------------------------------------------------------

class FineTuneSolarModel(SolarForecastModel):
    """SolarForecastModel with weather/generation encoders frozen.

    Inherits all forward/step logic from SolarForecastModel.  Only overrides
    configure_optimizers to restrict gradients to the trainable sub-modules:
    metadata_encoder, fusion, and head.

    The two sequence encoders are frozen because they encode the physical
    relationship between weather patterns and irradiance — this mapping
    is climate-universal and should transfer from TX/CA to NY without
    re-training.  Only the site-adaptation layers (metadata and head)
    need NY exposure to close the localisation gap.
    """

    def configure_optimizers(self):
        """Build AdamW over trainable (unfrozen) parameters only."""
        trainable = [p for p in self.parameters() if p.requires_grad]
        n_frozen  = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train   = sum(p.numel() for p in trainable)
        print(
            f"  Frozen parameters   : {n_frozen:,}\n"
            f"  Trainable parameters: {n_train:,}",
            flush=True,
        )
        opt = torch.optim.AdamW(
            trainable,
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
        return opt


# ---------------------------------------------------------------------------
# S3 checkpoint download
# ---------------------------------------------------------------------------

def download_s3_checkpoint(s3_uri: str) -> str:
    """Download an S3 checkpoint to a local temp file.

    Uses NamedTemporaryFile so Lightning can load it by path.  The caller is
    responsible for deleting the file when done (or relying on OS cleanup).

    Args:
        s3_uri: S3 URI of the form ``s3://bucket/key/file.ckpt``.

    Returns:
        Absolute local path to the downloaded temp file.

    Raises:
        ImportError: If boto3 is not installed.
        RuntimeError: If the download fails (credentials missing, key not found).
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 required: pip install boto3") from exc

    without_scheme = s3_uri[5:]
    bucket, _, key = without_scheme.partition("/")
    suffix = Path(key).suffix or ".ckpt"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()  # Close handle; boto3 needs to open it for writing.

    try:
        print(f"  Downloading checkpoint from S3 ...", flush=True)
        boto3.client("s3").download_file(bucket, key, tmp.name)
        size_mb = Path(tmp.name).stat().st_size / 1024 ** 2
        print(f"  Downloaded: {Path(key).name} ({size_mb:.1f} MB)", flush=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download checkpoint from {s3_uri}.\n"
            "Check your AWS credentials in .env:\n"
            "  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=us-east-2\n"
            f"Original error: {exc}"
        ) from exc

    return tmp.name


def resolve_ny_parquet(cfg: dict) -> str:
    """Return local path or S3 URI for test_ny.parquet (same logic as train.py)."""
    d      = cfg.get("data", {})
    local  = ROOT / d["test_ny_parquet"]
    if local.exists():
        return str(local)

    bucket = os.environ.get("S3_BUCKET") or d.get("s3_bucket", "")
    prefix = (
        os.environ.get("S3_PROCESSED_PREFIX")
        or os.environ.get("S3_DATA_PREFIX")
        or d.get("s3_data_prefix", "data/processed")
    )
    if bucket:
        filename = local.name
        return f"s3://{bucket}/{prefix.rstrip('/')}/{filename}"

    raise FileNotFoundError(
        f"test_ny.parquet not found locally: {local}\n"
        "Set S3_BUCKET in .env to stream from S3."
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_home_dataset(
    df: pl.DataFrame,
    dataid,
    seq_len: int,
    forecast_horizon: int,
    row_start: int = 0,
    row_end: Optional[int] = None,
) -> SolarWindowDataset:
    """Build a SolarWindowDataset for one home, optionally sliced by row range.

    Args:
        df:              Full DataFrame (all homes).
        dataid:          Home identifier to filter on.
        seq_len:         Context window length.
        forecast_horizon: Forecast steps.
        row_start:       First row index (inclusive).
        row_end:         Last row index (exclusive). None = all rows.

    Returns:
        SolarWindowDataset (may have zero windows if slice is too short).
    """
    home      = df.filter(pl.col("dataid") == dataid)
    meta_vals = [float(home.row(0, named=True)[c]) for c in METADATA_COLS]

    wx  = home.select(WEATHER_COLS).to_numpy().astype(np.float32)
    gen = home.select(SOLAR_COL).to_numpy().astype(np.float32)
    n   = min(len(wx), len(gen))
    end = n if row_end is None else min(row_end, n)

    ds = SolarWindowDataset.__new__(SolarWindowDataset)
    ds.weather          = torch.tensor(wx[row_start:end],  dtype=torch.float32)
    ds.generation       = torch.tensor(gen[row_start:end], dtype=torch.float32)
    ds.metadata         = torch.tensor(meta_vals,           dtype=torch.float32)
    ds.seq_len          = seq_len
    ds.forecast_horizon = forecast_horizon
    return ds


# ---------------------------------------------------------------------------
# Data modules
# ---------------------------------------------------------------------------

class FullNYDataModule(L.LightningDataModule):
    """Serves all NY home rows as the test split (for zero-shot baseline)."""

    def __init__(self, ny_local: str, cfg: dict, num_workers: int, pin_memory: bool):
        super().__init__()
        d = cfg["data"]
        self.ny_local         = ny_local
        self.seq_len          = d["seq_len"]
        self.forecast_horizon = d["forecast_horizon"]
        self.batch_size       = d["batch_size"]
        self.num_workers      = num_workers
        self.pin_memory       = pin_memory
        self._test_ds         = None

    def setup(self, stage: Optional[str] = None) -> None:
        ny         = read_parquet(self.ny_local)
        test_sets  = []
        for dataid in sorted(ny["dataid"].unique().to_list()):
            ds = _make_home_dataset(ny, dataid, self.seq_len, self.forecast_horizon)
            if len(ds) > 0:
                test_sets.append(ds)
        self._test_ds = ConcatDataset(test_sets)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class NYFineTuneDataModule(L.LightningDataModule):
    """Splits each NY home chronologically into fine-tune and held-out eval sets.

    Fine-tune split : first ``ny_days * STEPS_PER_DAY`` rows per home
    Eval split      : remaining rows per home

    The eval split is used as both validation (for early stopping) and final
    test (for reporting) — acceptable given the small amount of NY data
    available per experiment.
    """

    def __init__(
        self,
        ny_local: str,
        cfg: dict,
        ny_days: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        d = cfg["data"]
        self.ny_local         = ny_local
        self.ny_days          = ny_days
        self.seq_len          = d["seq_len"]
        self.forecast_horizon = d["forecast_horizon"]
        self.batch_size       = d["batch_size"]
        self.num_workers      = num_workers
        self.pin_memory       = pin_memory
        self.home_stats: list[dict] = []
        self._train_ds = None
        self._val_ds   = None
        self._test_ds  = None

    def setup(self, stage: Optional[str] = None) -> None:
        ny = read_parquet(self.ny_local)
        ft_sets:   list[SolarWindowDataset] = []
        eval_sets: list[SolarWindowDataset] = []
        self.home_stats = []

        for dataid in sorted(ny["dataid"].unique().to_list()):
            n_rows    = ny.filter(pl.col("dataid") == dataid).height
            ft_rows   = min(self.ny_days * STEPS_PER_DAY, n_rows)
            eval_start = ft_rows

            ft_ds   = _make_home_dataset(ny, dataid, self.seq_len, self.forecast_horizon,
                                          row_start=0,        row_end=ft_rows)
            eval_ds = _make_home_dataset(ny, dataid, self.seq_len, self.forecast_horizon,
                                          row_start=eval_start, row_end=None)

            ft_wins   = len(ft_ds)
            eval_wins = len(eval_ds)
            actual_days = ft_rows / STEPS_PER_DAY

            stat = {
                "dataid":     dataid,
                "n_rows":     n_rows,
                "ft_days":    f"{actual_days:.1f}",
                "ft_windows": ft_wins,
                "eval_windows": eval_wins,
                "warn":       "",
            }
            if ft_rows < self.ny_days * STEPS_PER_DAY:
                stat["warn"] = f"[WARN: only {actual_days:.0f} days available, requested {self.ny_days}]"
            if ft_wins == 0:
                stat["warn"] += " [SKIP: insufficient fine-tune data]"
            if eval_wins == 0:
                stat["warn"] += " [WARN: no eval data — home not in held-out test]"

            self.home_stats.append(stat)

            if ft_wins > 0:
                ft_sets.append(ft_ds)
            if eval_wins > 0:
                eval_sets.append(eval_ds)

        if not ft_sets:
            raise RuntimeError(
                f"No homes have enough data for {self.ny_days}-day fine-tuning. "
                "Try a smaller --ny_days value."
            )

        self._train_ds = ConcatDataset(ft_sets)
        # Use eval split as both val (early stopping) and test (final reporting)
        self._val_ds   = ConcatDataset(eval_sets) if eval_sets else ConcatDataset(ft_sets)
        self._test_ds  = self._val_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# (Results logging moved to src/results_utils.py — save_row / save_per_home)


# ---------------------------------------------------------------------------
# Hardware detection (mirror train.py)
# ---------------------------------------------------------------------------

def detect_hardware():
    """Return (hw_name, precision, pin_memory) based on available hardware."""
    cuda_available = torch.cuda.is_available()
    try:
        mps_available = torch.backends.mps.is_available()
    except AttributeError:
        mps_available = False

    if cuda_available:
        return f"CUDA ({torch.cuda.get_device_name(0)})", "16-mixed", True
    if mps_available:
        return "MPS (Apple Silicon)", "32-true", False
    return "CPU", "32-true", False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune solar forecasting model on NY data.",
    )
    parser.add_argument(
        "--ny_days", type=int, required=True,
        help="Days of NY data to use for fine-tuning (any positive integer).",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="S3 URI or local path to base checkpoint.",
    )
    parser.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment YAML config (relative to repo root).",
    )
    parser.add_argument(
        "--model_version", default="v1",
        help="Model version tag used for checkpoint naming and results (e.g. v1, v2).",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    t = cfg["trainer"]

    # -- Setup ----------------------------------------------------------------
    num_workers = min(os.cpu_count() or 1, 8)
    hw_name, precision, pin_memory = detect_hardware()

    s3_bucket      = os.environ.get("S3_BUCKET", "")
    s3_ckpt_prefix = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")
    ft_run_name    = f"finetune_{args.model_version}_ny{args.ny_days}days"
    s3_ft_prefix   = f"{s3_ckpt_prefix}/{ft_run_name}" if s3_bucket else ""

    print(f"\n{'='*65}")
    print("  FINE-TUNING RUN")
    print(f"{'='*65}")
    print(f"  Config          : {args.config}")
    print(f"  Base checkpoint : {args.checkpoint}")
    print(f"  Model version   : {args.model_version}")
    print(f"  NY fine-tune    : {args.ny_days} days per home")
    print(f"  Hardware        : {hw_name}")
    print(f"  Precision       : {precision}")
    if s3_bucket:
        print(f"  S3 FT checkpts  : s3://{s3_bucket}/{s3_ft_prefix}/")

    # -- Download base checkpoint ---------------------------------------------
    print(f"\n[1/5] Loading base checkpoint")
    ckpt_str = args.checkpoint
    if ckpt_str.startswith("s3://"):
        ckpt_local = download_s3_checkpoint(ckpt_str)
    else:
        ckpt_local = str(ROOT / ckpt_str) if not Path(ckpt_str).is_absolute() else ckpt_str
        if not Path(ckpt_local).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_local}")

    from src.models import build as _build_model
    base_model = _build_model(config_path)
    ckpt_data  = torch.load(ckpt_local, map_location="cpu", weights_only=False)
    base_model.load_state_dict(ckpt_data["state_dict"])
    base_model.eval()
    print(f"  Loaded checkpoint: {Path(ckpt_local).name}")

    # -- Load NY data ---------------------------------------------------------
    print(f"\n[2/5] Loading NY data")
    ny_path  = resolve_ny_parquet(cfg)
    cache_dir = ROOT / "data" / "processed"
    ny_local  = ensure_local_parquet(ny_path, cache_dir)

    # -- Zero-shot baseline ---------------------------------------------------
    print(f"\n[3/5] Zero-shot baseline evaluation (full NY, all homes)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fh = cfg["data"]["forecast_horizon"]

    zs_metrics, zs_per_home = evaluate_parquet(base_model, ny_local, cfg, device=device)
    zero_shot_mae = zs_metrics["mae"]
    print(f"  Zero-shot MAE: {zero_shot_mae:.4f} kWh  R²={zs_metrics['r2']:.3f}  Skill={zs_metrics['skill_score']:.1f}%")

    # -- Fine-tune data split -------------------------------------------------
    print(f"\n[4/5] Preparing fine-tune / eval split ({args.ny_days}-day window per home)")
    ft_dm = NYFineTuneDataModule(
        ny_local, cfg, ny_days=args.ny_days,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    ft_dm.setup()

    # Print per-home stats
    print(f"\n  {'dataid':<10} {'ft_days':>8} {'ft_windows':>11} {'eval_windows':>13}  note")
    print(f"  {'-'*65}")
    total_ft = total_eval = 0
    for s in ft_dm.home_stats:
        print(
            f"  {str(s['dataid']):<10} {s['ft_days']:>8} "
            f"{s['ft_windows']:>11,} {s['eval_windows']:>13,}  {s['warn']}",
            flush=True,
        )
        total_ft   += s["ft_windows"]
        total_eval += s["eval_windows"]
    print(f"  {'-'*65}")
    print(f"  {'TOTAL':<10} {'':>8} {total_ft:>11,} {total_eval:>13,}")

    # -- Build fine-tune model ------------------------------------------------
    # Construct FineTuneSolarModel sharing the same module objects as base.
    # Freezing is done in-place on the shared references.
    ft_model = FineTuneSolarModel(
        weather_encoder=base_model.weather_encoder,
        generation_encoder=base_model.generation_encoder,
        metadata_encoder=base_model.metadata_encoder,
        fusion=base_model.fusion,
        head=base_model.head,
        lr=FT_LR,
        weight_decay=base_model.hparams.weight_decay,
        scheduler=base_model.hparams.scheduler,
        epochs=50,
    )
    # Freeze the two sequence encoders
    ft_model.weather_encoder.requires_grad_(False)
    ft_model.generation_encoder.requires_grad_(False)

    # -- Fine-tune training ---------------------------------------------------
    print(f"\n[5/5] Fine-tuning (lr={FT_LR}, max_epochs=50, patience=5)")

    ckpt_dir = ROOT / "models" / "checkpoints" / ft_run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = S3ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"ft-{args.model_version}-ny{args.ny_days}d-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        s3_bucket=s3_bucket,
        s3_prefix=s3_ft_prefix,
    )
    early_stop_cb = VerboseEarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=False,
    )
    epoch_bar = EpochSummaryCallback(max_epochs=50)
    logger    = CSVLogger(save_dir=str(ROOT / "logs"), name=f"finetune_{args.model_version}")

    ft_trainer = L.Trainer(
        max_epochs=50,
        accelerator=t["accelerator"],
        devices=t["devices"],
        precision=precision,
        log_every_n_steps=50,
        val_check_interval=1.0,
        gradient_clip_val=t.get("gradient_clip_val", 1.0),
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint_cb, early_stop_cb, epoch_bar],
        logger=logger,
        default_root_dir=str(ROOT / "logs"),
    )

    ft_trainer.fit(ft_model, datamodule=ft_dm)
    epoch_stopped = ft_trainer.current_epoch + 1

    # -- Post-fine-tune evaluation on held-out eval set -----------------------
    best_ckpt = checkpoint_cb.best_model_path or "best"
    if best_ckpt != "best" and Path(best_ckpt).exists():
        best_ft_model = _build_model(config_path)
        ft_ckpt_data  = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        best_ft_model.load_state_dict(ft_ckpt_data["state_dict"])
    else:
        best_ft_model = ft_model
    best_ft_model.eval()

    ft_metrics = evaluate_from_loader(
        best_ft_model, ft_dm.test_dataloader(), device=device, forecast_horizon=fh
    )
    finetuned_mae  = ft_metrics["mae"]
    improvement    = zero_shot_mae - finetuned_mae
    pct_gap_closed = (improvement / zero_shot_mae * 100) if zero_shot_mae > 0 else 0.0

    # -- S3 checkpoint path for logging ---------------------------------------
    if s3_bucket and checkpoint_cb.best_model_path:
        best_fname   = Path(checkpoint_cb.best_model_path).name
        ckpt_s3_path = f"s3://{s3_bucket}/{s3_ft_prefix}/{best_fname}"
    elif not ckpt_str.startswith("s3://"):
        ckpt_s3_path = checkpoint_cb.best_model_path or "local"
    else:
        ckpt_s3_path = f"s3://{s3_bucket}/{s3_ft_prefix}/" if s3_bucket else "local"

    experiment_name = f"finetune_{args.ny_days}d"

    # -- Print results --------------------------------------------------------
    print(f"\n{'='*65}")
    print("  FINE-TUNING RESULTS")
    print(f"{'='*65}")
    print(f"  Zero-shot baseline MAE    : {zero_shot_mae:.4f} kWh")
    print(f"  Fine-tuned MAE ({args.ny_days:2d} days)  : {finetuned_mae:.4f} kWh")
    print(f"  Fine-tuned RMSE           : {ft_metrics['rmse']:.4f} kWh")
    print(f"  Fine-tuned R²             : {ft_metrics['r2']:.4f}")
    print(f"  Fine-tuned Skill score    : {ft_metrics['skill_score']:.2f}%")
    print(f"  Absolute improvement      : {improvement:+.4f} kWh")
    print(f"  Gap closed                : {pct_gap_closed:.1f}%")
    print(f"  Early stopping epoch      : {epoch_stopped}")
    print(f"  Best checkpoint           : {checkpoint_cb.best_model_path or 'N/A'}")
    if s3_bucket:
        print(f"  S3 path                   : {ckpt_s3_path}")
    print(f"{'='*65}\n")

    # -- Save results ---------------------------------------------------------
    import math as _math
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Zero-shot row (saves baseline for comparison)
    save_row({
        "model_version":      args.model_version,
        "experiment":         "zero_shot",
        "ny_days":            0,
        **{k: ("" if isinstance(v, float) and _math.isnan(v) else v) for k, v in zs_metrics.items()},
        "generalization_gap": "",
        "epoch_stopped":      "",
        "timestamp":          now_str,
        "checkpoint_s3_path": ckpt_str,
    })

    # Fine-tuned row
    save_row({
        "model_version":      args.model_version,
        "experiment":         experiment_name,
        "ny_days":            args.ny_days,
        **{k: ("" if isinstance(v, float) and _math.isnan(v) else v) for k, v in ft_metrics.items()},
        "generalization_gap": "",
        "epoch_stopped":      epoch_stopped,
        "timestamp":          now_str,
        "checkpoint_s3_path": ckpt_s3_path,
    })

    # Per-home from zero-shot
    save_per_home([
        {"model_version": args.model_version, "experiment": "zero_shot",
         "ny_days": 0, "timestamp": now_str, **h}
        for h in zs_per_home
    ])

    auto_commit(
        mae=finetuned_mae,
        r2=ft_metrics["r2"] if not _math.isnan(ft_metrics["r2"]) else 0.0,
        skill=ft_metrics["skill_score"],
        model_version=args.model_version,
        experiment=experiment_name,
    )

    # Cleanup temp checkpoint if we downloaded from S3
    if args.checkpoint.startswith("s3://"):
        try:
            Path(ckpt_local).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
