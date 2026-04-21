"""src/evaluate.py — Standalone evaluation script for any checkpoint + parquet.

Computes all 7 metrics: MAE, RMSE, MAPE, R², skill score vs persistence,
peak-hour MAE (8am–4pm), and generalization gap.  Also saves per-home metrics
and appends to results/all_results.csv.

Usage
-----
    # Re-evaluate V1 zero-shot on NY test set:
    python src/evaluate.py \\
        --checkpoint s3://cs7180-final-project/checkpoints/2026-04-21_01-40-26/solar-epoch=04-val_loss=0.0042.ckpt \\
        --data data/processed/test_ny.parquet \\
        --output_name zero_shot \\
        --model_version v1

    # Evaluate on in-region TX training data:
    python src/evaluate.py \\
        --checkpoint s3://...ckpt \\
        --data data/processed/train_tx.parquet \\
        --output_name in_region_tx \\
        --model_version v1
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    for _raw in _env.read_text().splitlines():
        _raw = _raw.strip()
        if _raw and not _raw.startswith("#") and "=" in _raw:
            _k, _v = _raw.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import polars as pl
from torch.utils.data import DataLoader

from src.dataloader import SolarWindowDataset, ensure_local_parquet, read_parquet
from src.metrics import compute_all, compute_mae, compute_r2
from src.models import SolarForecastModel
from src.results_utils import auto_commit, generate_markdown, save_per_home, save_row

WEATHER_COLS  = ["GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct"]
SOLAR_COL     = "solar_kwh"
METADATA_COLS = ["lat", "lon", "tilt_deg", "azimuth_deg", "capacity_kw", "elevation_m"]
TIMESTAMP_COL = "local_15min"


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _download_s3(s3_uri: str) -> str:
    """Download S3 checkpoint to a temp file and return the local path."""
    import boto3
    without_scheme = s3_uri[5:]
    bucket, _, key = without_scheme.partition("/")
    suffix = Path(key).suffix or ".ckpt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    print(f"  Downloading checkpoint: {Path(key).name}", flush=True)
    boto3.client("s3").download_file(bucket, key, tmp.name)
    size_mb = Path(tmp.name).stat().st_size / 1024 ** 2
    print(f"  Downloaded: {size_mb:.1f} MB", flush=True)
    return tmp.name


def resolve_checkpoint(checkpoint: str) -> tuple[str, bool]:
    """Return (local_path, is_temp).  is_temp=True means caller should delete it."""
    if not checkpoint.startswith("s3://"):
        local = ROOT / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        if not local.exists():
            raise FileNotFoundError(f"Checkpoint not found: {local}")
        return str(local), False

    # Check local models/checkpoints/ for a cached copy
    filename = Path(checkpoint).name
    for candidate in (ROOT / "models" / "checkpoints").rglob("*.ckpt"):
        if candidate.name == filename:
            print(f"  Using local checkpoint fallback: {candidate.relative_to(ROOT)}", flush=True)
            return str(candidate), False

    # Download from S3
    try:
        return _download_s3(checkpoint), True
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {checkpoint}\n"
            "Check AWS credentials in .env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------

def resolve_parquet(data_arg: str, cfg: dict) -> str:
    """Return a local path to the parquet, downloading from S3 if needed."""
    # Absolute or relative local path
    local = Path(data_arg)
    if not local.is_absolute():
        local = ROOT / data_arg
    if local.exists():
        return str(local)

    # S3 URI
    if data_arg.startswith("s3://"):
        cache_dir = ROOT / "data" / "processed"
        return ensure_local_parquet(data_arg, cache_dir)

    # Try building S3 URI from config
    d      = cfg.get("data", {})
    bucket = os.environ.get("S3_BUCKET") or d.get("s3_bucket", "")
    prefix = (
        os.environ.get("S3_PROCESSED_PREFIX")
        or os.environ.get("S3_DATA_PREFIX")
        or d.get("s3_data_prefix", "data/processed")
    )
    if bucket:
        filename = Path(data_arg).name
        s3_uri   = f"s3://{bucket}/{prefix.rstrip('/')}/{filename}"
        cache_dir = ROOT / "data" / "processed"
        return ensure_local_parquet(s3_uri, cache_dir)

    raise FileNotFoundError(
        f"Parquet not found: {data_arg}\n"
        "Set S3_BUCKET in .env or provide an absolute path."
    )


# ---------------------------------------------------------------------------
# Core evaluation logic — importable by train.py and finetune.py
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_from_loader(
    model: SolarForecastModel,
    loader: DataLoader,
    device: str = "cpu",
    forecast_horizon: int = 4,
) -> dict:
    """Evaluate model on any DataLoader; returns metrics dict (peak_mae=NaN).

    Persistence is computed from each batch's generation context window
    (first forecast_horizon steps = same time yesterday for 24-h context).
    """
    model.eval()
    model.to(device)
    all_preds, all_actuals, all_persistence = [], [], []

    for weather, gen, meta, target in loader:
        weather, gen, meta = weather.to(device), gen.to(device), meta.to(device)
        preds = model(weather, gen, meta)
        all_preds.append(preds.cpu().numpy())
        all_actuals.append(target.numpy())
        all_persistence.append(gen[:, :forecast_horizon, 0].cpu().numpy())

    preds_np  = np.concatenate(all_preds,       axis=0)
    actual_np = np.concatenate(all_actuals,      axis=0)
    pers_np   = np.concatenate(all_persistence,  axis=0)

    return compute_all(preds_np, actual_np, persistence=pers_np)


@torch.no_grad()
def evaluate_parquet(
    model: SolarForecastModel,
    parquet_local: str,
    cfg: dict,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[dict, list[dict]]:
    """Evaluate model on a parquet file with full 7-metric suite and per-home breakdown.

    Returns
    -------
    (aggregate_metrics, per_home_records)
    - aggregate_metrics : dict with mae, rmse, mape, r2, skill_score, peak_mae
    - per_home_records  : list of dicts with dataid, mae, r2, n_windows per home
    """
    model.eval()
    model.to(device)

    df  = read_parquet(parquet_local)
    seq = cfg["data"]["seq_len"]
    fh  = cfg["data"]["forecast_horizon"]
    has_ts = TIMESTAMP_COL in df.columns

    all_preds, all_actuals, all_persistence, all_hours = [], [], [], []
    per_home_records: list[dict] = []

    for dataid in sorted(df["dataid"].unique().to_list()):
        home_df = df.filter(pl.col("dataid") == dataid)
        n = home_df.height

        if n < seq + fh:
            continue

        wx  = home_df.select(WEATHER_COLS).to_numpy().astype(np.float32)
        gen = home_df.select(SOLAR_COL).to_numpy().astype(np.float32).reshape(-1, 1)
        meta_vals = [float(home_df.row(0, named=True)[c]) for c in METADATA_COLS]

        # Timestamps: local hour-of-day per row
        hour_arr = None
        if has_ts:
            ts_list = home_df[TIMESTAMP_COL].to_list()
            try:
                if ts_list and hasattr(ts_list[0], "hour"):
                    # Python datetime objects (Polars may return these)
                    hour_arr = np.array([t.hour for t in ts_list], dtype=np.int32)
                else:
                    # String timestamps "YYYY-MM-DD HH:MM:SS±offset"
                    # Slice "HH" from position 11:13 — avoids mixed-tz pandas issues
                    hour_arr = np.array([int(str(t)[11:13]) for t in ts_list], dtype=np.int32)
            except Exception:
                hour_arr = None

        # Build dataset
        ds = SolarWindowDataset.__new__(SolarWindowDataset)
        ds.weather          = torch.tensor(wx,  dtype=torch.float32)
        ds.generation       = torch.tensor(gen, dtype=torch.float32)
        ds.metadata         = torch.tensor(meta_vals, dtype=torch.float32)
        ds.seq_len          = seq
        ds.forecast_horizon = fh

        if len(ds) == 0:
            continue

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        home_preds, home_actuals, home_persistence = [], [], []
        for weather_b, gen_b, meta_b, target_b in loader:
            weather_b, gen_b, meta_b = weather_b.to(device), gen_b.to(device), meta_b.to(device)
            p = model(weather_b, gen_b, meta_b)
            home_preds.append(p.cpu().numpy())
            home_actuals.append(target_b.numpy())
            home_persistence.append(gen_b[:, :fh, 0].cpu().numpy())

        hp  = np.concatenate(home_preds,       axis=0)  # (n_windows, fh)
        ha  = np.concatenate(home_actuals,      axis=0)
        hpe = np.concatenate(home_persistence,  axis=0)

        # Hours per forecast step for this home
        if hour_arr is not None:
            n_windows = len(ds)
            win_starts = np.arange(n_windows)
            idx_matrix = win_starts[:, None] + seq + np.arange(fh)[None, :]  # (n_windows, fh)
            idx_matrix = np.clip(idx_matrix, 0, n - 1)
            home_hours = hour_arr[idx_matrix].flatten()  # (n_windows * fh,)
        else:
            home_hours = None

        all_preds.append(hp)
        all_actuals.append(ha)
        all_persistence.append(hpe)
        if home_hours is not None:
            all_hours.append(home_hours)

        # Per-home metrics
        home_m = compute_all(hp, ha, persistence=hpe, hours=home_hours)
        per_home_records.append({
            "dataid":    dataid,
            "mae":       home_m["mae"],
            "r2":        home_m["r2"],
            "n_windows": len(ds),
        })

    preds_all  = np.concatenate(all_preds,      axis=0)
    actual_all = np.concatenate(all_actuals,     axis=0)
    pers_all   = np.concatenate(all_persistence, axis=0)
    hours_all  = np.concatenate(all_hours, axis=0) if all_hours else None

    agg = compute_all(preds_all, actual_all, persistence=pers_all, hours=hours_all)
    return agg, per_home_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on any parquet file.")
    parser.add_argument("--checkpoint", required=True,
                        help="S3 URI or local path to .ckpt file.")
    parser.add_argument("--data", required=True,
                        help="Parquet path (local or S3 URI) to evaluate on.")
    parser.add_argument("--split", default="test",
                        help="Split label for display (train/val/test). Default: test.")
    parser.add_argument("--output_name", required=True,
                        help="experiment column value (e.g. zero_shot, in_region_tx).")
    parser.add_argument("--model_version", default="v1",
                        help="Model version tag (v1, v2, …).")
    parser.add_argument("--ny_days", type=int, default=0,
                        help="Fine-tune days used (0 = zero-shot).")
    parser.add_argument("--config", default="configs/experiment.yaml",
                        help="Path to experiment YAML (relative to repo root).")
    parser.add_argument("--in_region_mae", type=float, default=None,
                        help="In-region MAE to compute generalisation gap against.")
    parser.add_argument("--no_commit", action="store_true",
                        help="Skip git auto-commit after evaluation.")
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # -- Hardware --
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except AttributeError:
        pass
    print(f"\n  Device : {device}")

    # -- Checkpoint --
    print(f"\n[1/3] Loading checkpoint: {args.checkpoint}")
    ckpt_local, is_temp = resolve_checkpoint(args.checkpoint)
    from src.models import build
    model = build(config_path)
    ckpt_data = torch.load(ckpt_local, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_data["state_dict"])
    model.eval()
    print(f"  Loaded: {Path(ckpt_local).name}")

    # -- Data --
    print(f"\n[2/3] Loading parquet: {args.data}")
    parquet_local = resolve_parquet(args.data, cfg)
    df_check = read_parquet(parquet_local)
    print(f"  Homes : {df_check['dataid'].n_unique()}")
    print(f"  Rows  : {df_check.height:,}")

    # -- Evaluate --
    print(f"\n[3/3] Evaluating ({args.split} split)…")
    metrics, per_home = evaluate_parquet(model, parquet_local, cfg, device=device)

    gen_gap = (
        metrics["mae"] - args.in_region_mae
        if args.in_region_mae is not None else float("nan")
    )

    # Determine checkpoint S3 path for logging
    ckpt_s3 = args.checkpoint if args.checkpoint.startswith("s3://") else ckpt_local

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {args.model_version} / {args.output_name}")
    print(f"{'='*60}")
    for name, val in metrics.items():
        print(f"  {name:<20}: {val:.4f}" if not (isinstance(val, float) and val != val) else f"  {name:<20}: NaN")
    if args.in_region_mae is not None:
        print(f"  {'generalization_gap':<20}: {gen_gap:+.4f}")
    print(f"\n  Per-home summary ({len(per_home)} homes):")
    for h in per_home:
        print(f"    dataid={h['dataid']}  MAE={h['mae']:.4f}  R²={h['r2']:.3f}  windows={h['n_windows']:,}")
    print(f"{'='*60}\n")

    # -- Save --
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    import math
    row = {
        "model_version":      args.model_version,
        "experiment":         args.output_name,
        "ny_days":            args.ny_days,
        **{k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()},
        "generalization_gap": ("" if math.isnan(gen_gap) else gen_gap),
        "epoch_stopped":      "",
        "timestamp":          now,
        "checkpoint_s3_path": ckpt_s3,
    }
    save_row(row)

    per_home_rows = [
        {
            "model_version": args.model_version,
            "experiment":    args.output_name,
            "ny_days":       args.ny_days,
            "timestamp":     now,
            **h,
        }
        for h in per_home
    ]
    save_per_home(per_home_rows)

    if not args.no_commit:
        import math as _math
        auto_commit(
            mae=metrics["mae"],
            r2=metrics["r2"] if not _math.isnan(metrics["r2"]) else 0.0,
            skill=metrics["skill_score"] if not _math.isnan(metrics["skill_score"]) else float("nan"),
            model_version=args.model_version,
            experiment=args.output_name,
        )

    # Cleanup temp checkpoint
    if is_temp:
        try:
            Path(ckpt_local).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
