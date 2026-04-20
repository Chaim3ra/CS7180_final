"""Training entry point for the multi-modal solar generation forecasting model.

Reads TX + CA processed parquets for training/validation and NY parquet for
zero-shot transfer evaluation.  Parquets are loaded from local disk when
present; if local files are absent and S3_BUCKET is configured, they are
streamed directly from S3 via boto3 — no dvc pull needed.

Usage:
    python src/train.py
    python src/train.py --config configs/experiment.yaml
    python src/train.py --fast   # 5 epochs, smoke-test only

S3 setup (for teammates without local data):
    Set in .env:
        AWS_ACCESS_KEY_ID=...
        AWS_SECRET_ACCESS_KEY=...
        AWS_DEFAULT_REGION=us-east-2
        S3_BUCKET=cs7180-final-project
        S3_DATA_PREFIX=data/processed   # optional, this is the default
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for raw in env_path.read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]
    from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore[no-redef]
    from pytorch_lightning.loggers import CSVLogger          # type: ignore[no-redef]

from src.models import build
from src.dataloader import SolarWindowDataset, read_parquet

WEATHER_COLS  = ["GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct"]
SOLAR_COL     = "solar_kwh"
METADATA_COLS = ["lat", "lon", "tilt_deg", "azimuth_deg", "capacity_kw", "elevation_m"]


# ---------------------------------------------------------------------------
# Path resolution: local-first, S3 fallback
# ---------------------------------------------------------------------------

def _resolve_parquet(local_rel: str, cfg: dict) -> str:
    """Return the path (local or S3 URI) to use for a processed parquet file.

    Priority:
      1. Local file at ``ROOT/local_rel`` — use it if it exists.
      2. S3 URI constructed from S3_BUCKET env var (or config) — if set.
      3. Raise ``FileNotFoundError`` with setup instructions.

    Args:
        local_rel: Relative path from repo root, e.g. ``data/processed/train_tx.parquet``.
        cfg:       Loaded experiment config dict.

    Returns:
        Absolute local path string or ``s3://bucket/prefix/filename`` URI.
    """
    local = ROOT / local_rel
    if local.exists():
        return str(local)

    # S3 fallback: env var takes precedence over config
    d      = cfg.get("data", {})
    bucket = os.environ.get("S3_BUCKET") or d.get("s3_bucket", "")
    prefix = os.environ.get("S3_DATA_PREFIX") or d.get("s3_data_prefix", "data/processed")

    if bucket:
        filename = Path(local_rel).name
        s3_uri   = f"s3://{bucket}/{prefix.rstrip('/')}/{filename}"
        return s3_uri

    raise FileNotFoundError(
        f"Parquet not found locally: {local}\n"
        f"Either run `dvc pull` to download data locally, or set S3_BUCKET in .env "
        f"to stream directly from S3.\n"
        f"  S3_BUCKET=cs7180-final-project\n"
        f"  S3_DATA_PREFIX=data/processed  # default if omitted"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap(dataset, max_windows):
    """Return a Subset of the first max_windows samples, or the full dataset if None."""
    if max_windows is None or len(dataset) <= max_windows:
        return dataset
    return Subset(dataset, list(range(max_windows)))


# ---------------------------------------------------------------------------
# Per-home dataset builder
# ---------------------------------------------------------------------------

def _make_home_dataset(
    df: pl.DataFrame,
    dataid: int,
    seq_len: int,
    forecast_horizon: int,
    row_start: int = 0,
    row_end: Optional[int] = None,
) -> SolarWindowDataset:
    """Build a SolarWindowDataset for one home, optionally sliced by row range."""
    home = df.filter(pl.col("dataid") == dataid)
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
# LightningDataModule
# ---------------------------------------------------------------------------

class MultiHomeDataModule(L.LightningDataModule):
    """LightningDataModule for multi-home parquet data.

    - Train/val: TX + CA parquets, split 85/15 per home (chronological).
    - Test:      NY parquet — all rows per home (zero-shot).
    """

    def __init__(self, cfg: dict, fast: bool = False):
        super().__init__()
        d = cfg["data"]
        # Resolve each parquet: local path if it exists, S3 URI otherwise.
        self.tx_path          = _resolve_parquet(d["train_tx_parquet"], cfg)
        self.ca_path          = _resolve_parquet(d["train_ca_parquet"], cfg)
        self.ny_path          = _resolve_parquet(d["test_ny_parquet"],  cfg)
        self.seq_len          = d["seq_len"]
        self.forecast_horizon = d["forecast_horizon"]
        self.train_frac       = d["train_frac"]
        self.batch_size       = d["batch_size"]
        self.num_workers      = d["num_workers"]
        # In fast mode cap windows so CPU smoke-test finishes in seconds.
        self.fast_train_cap = 2_000 if fast else None
        self.fast_val_cap   =   500 if fast else None
        self.fast_test_cap  =   500 if fast else None

        self._train_ds = None
        self._val_ds   = None
        self._test_ds  = None

    def setup(self, stage: Optional[str] = None) -> None:
        # read_parquet handles both local paths and s3:// URIs transparently
        tx = read_parquet(self.tx_path)
        ca = read_parquet(self.ca_path)
        ny = read_parquet(self.ny_path)

        train_sets: list[SolarWindowDataset] = []
        val_sets:   list[SolarWindowDataset] = []
        for df in (tx, ca):
            for dataid in sorted(df["dataid"].unique().to_list()):
                n_rows = df.filter(pl.col("dataid") == dataid).height
                split  = int(n_rows * self.train_frac)
                tr = _make_home_dataset(df, dataid, self.seq_len, self.forecast_horizon,
                                        row_start=0,     row_end=split)
                va = _make_home_dataset(df, dataid, self.seq_len, self.forecast_horizon,
                                        row_start=split, row_end=None)
                if len(tr) > 0:
                    train_sets.append(tr)
                if len(va) > 0:
                    val_sets.append(va)

        test_sets: list[SolarWindowDataset] = []
        for dataid in sorted(ny["dataid"].unique().to_list()):
            ds = _make_home_dataset(ny, dataid, self.seq_len, self.forecast_horizon)
            if len(ds) > 0:
                test_sets.append(ds)

        full_train = ConcatDataset(train_sets)
        full_val   = ConcatDataset(val_sets)
        full_test  = ConcatDataset(test_sets)

        self._train_ds = _cap(full_train, self.fast_train_cap)
        self._val_ds   = _cap(full_val,   self.fast_val_cap)
        self._test_ds  = _cap(full_test,  self.fast_test_cap)

    def train_dataloader(self) -> DataLoader:
        pin = torch.cuda.is_available()
        return DataLoader(self._train_ds, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers, pin_memory=pin)

    def val_dataloader(self) -> DataLoader:
        pin = torch.cuda.is_available()
        return DataLoader(self._val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=pin)

    def test_dataloader(self) -> DataLoader:
        pin = torch.cuda.is_available()
        return DataLoader(self._test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=pin)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/experiment.yaml", fast: bool = False) -> None:
    config_path = ROOT / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # -- Data -----------------------------------------------------------------
    dm = MultiHomeDataModule(cfg, fast=fast)
    dm.setup()

    print(f"\n{'='*60}")
    print("  TRAINING RUN")
    print(f"{'='*60}")
    print(f"  Config           : {config_path.relative_to(ROOT)}")
    fast_tag = " [capped]" if fast else ""
    print(f"  Train windows    (TX+CA 85%) : {len(dm._train_ds):>8,}{fast_tag}")
    print(f"  Val   windows    (TX+CA 15%) : {len(dm._val_ds):>8,}{fast_tag}")
    print(f"  Test  windows    (NY  100%)  : {len(dm._test_ds):>8,}{fast_tag}")

    # -- Model ----------------------------------------------------------------
    model = build(config_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters             : {n_params:>8,}")

    # -- Checkpoint & logger --------------------------------------------------
    ckpt_dir = ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="solar-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    logger = CSVLogger(save_dir=str(ROOT / "logs"), name="solar_forecast")

    # -- Trainer --------------------------------------------------------------
    t = cfg["trainer"]
    max_epochs = 5 if fast else t["max_epochs"]
    if fast:
        print(f"  [FAST MODE] max_epochs=5, train cap=2000, val cap=500, test cap=500")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=t["accelerator"],
        devices=t["devices"],
        precision=t["precision"],
        log_every_n_steps=t["log_every_n_steps"],
        val_check_interval=t["val_check_interval"],
        gradient_clip_val=t["gradient_clip_val"],
        enable_progress_bar=t["enable_progress_bar"],
        enable_model_summary=t["enable_model_summary"],
        callbacks=[checkpoint_cb],
        logger=logger,
        default_root_dir=str(ROOT / "logs"),
    )
    print(f"{'='*60}\n")

    # -- Train ----------------------------------------------------------------
    trainer.fit(model, datamodule=dm)

    # -- Post-training evaluation --------------------------------------------
    print(f"\n{'='*60}")
    print("  POST-TRAINING EVALUATION")
    print(f"{'='*60}")

    best_ckpt = checkpoint_cb.best_model_path or "best"

    val_results  = trainer.validate(model, datamodule=dm, ckpt_path=best_ckpt, verbose=False)
    test_results = trainer.test(model,     datamodule=dm, ckpt_path=best_ckpt, verbose=False)

    in_mae  = val_results[0]["val_mae"]
    out_mae = test_results[0]["test_mae"]
    gap     = out_mae - in_mae

    print(f"  In-region  MAE  (TX+CA val)   : {in_mae:.4f} kWh")
    print(f"  Out-region MAE  (NY zero-shot) : {out_mae:.4f} kWh")
    print(f"  Generalization gap             : {gap:+.4f} kWh")
    print(f"\n  Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"  CSV logs        : {logger.log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train solar generation forecasting model.")
    parser.add_argument("--config", default="configs/experiment.yaml",
                        help="Path to experiment YAML (relative to repo root).")
    parser.add_argument("--fast", action="store_true",
                        help="Run 5 epochs only (smoke test).")
    args = parser.parse_args()
    main(config_path=args.config, fast=args.fast)
