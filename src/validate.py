"""End-to-end validation: data pipeline, preprocessing, windowing, model, splits.

Exit 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# ── Resolve repo root and load .env ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

env_path = ROOT / ".env"
if env_path.exists():
    for raw in env_path.read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_dr = os.environ.get("DATA_ROOT", "data/raw")
DATA_ROOT = Path(_dr) if Path(_dr).is_absolute() else ROOT / _dr

sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.models import build
from src.dataloader import SolarWindowDataset

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN          = 96
FORECAST_HORIZON = 4
BATCH            = 8
N_WEATHER        = 6
N_METADATA       = 6
WEATHER_COLS     = ["GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct"]

# Representative metadata per region: lat, lon, tilt, azimuth, capacity_kw, elevation_m
REGION_META = {
    "tx": [30.2672, -97.7431, 20.0, 180.0, 5.0, 200.0],
    "ca": [37.3382, -121.8863, 20.0, 180.0, 5.0, 30.0],
    "ny": [40.7128, -74.0060,  20.0, 180.0, 5.0, 10.0],
}

RESULTS: dict[str, bool] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(n: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  CHECK {n} — {title}")
    print(f"{'='*60}")


def mark(name: str, passed: bool, msg: str = "") -> None:
    RESULTS[name] = passed
    tag = "PASS" if passed else "FAIL"
    suffix = f" — {msg}" if msg else ""
    print(f"  [{tag}] {name}{suffix}")


def homes_with_solar(df: pl.DataFrame) -> list[int]:
    return (
        df.filter(pl.col("solar").is_not_null())
          .select("dataid").unique().sort("dataid")
          ["dataid"].to_list()
    )


def align_weather_solar(
    solar_df: pl.DataFrame,
    weather_df: pl.DataFrame,
    dataid: int,
) -> Optional[pl.DataFrame]:
    """Left-join hourly NASA-POWER weather to 15-min solar rows for one home.

    Uses the first 13 chars of the timestamp string (YYYY-MM-DD HH) as the
    join key so timezone offsets in the Pecan Street timestamps are ignored
    without a full datetime parse.
    """
    home = (
        solar_df
        .filter(pl.col("dataid") == dataid)
        .select(["local_15min", "solar"])
        .with_columns(
            pl.col("local_15min").str.slice(0, 13).alias("ts_hour")
        )
    )
    wx = (
        weather_df
        .with_columns(pl.col("datetime").str.slice(0, 13).alias("ts_hour"))
        .select(["ts_hour"] + WEATHER_COLS)
    )
    joined = home.join(wx, on="ts_hour", how="left")
    # Keep rows where solar and all weather cols are non-null
    joined = joined.filter(pl.col("solar").is_not_null())
    for col in WEATHER_COLS:
        joined = joined.filter(pl.col(col).is_not_null())
    return joined if joined.height >= SEQ_LEN + FORECAST_HORIZON else None


def make_dataset(aligned: pl.DataFrame, region: str) -> SolarWindowDataset:
    wx  = aligned.select(WEATHER_COLS).to_numpy().astype(np.float32)
    gen = aligned.select("solar").to_numpy().astype(np.float32)
    n   = min(len(wx), len(gen))
    ds  = SolarWindowDataset.__new__(SolarWindowDataset)
    ds.weather          = torch.tensor(wx[:n],  dtype=torch.float32)
    ds.generation       = torch.tensor(gen[:n], dtype=torch.float32)
    ds.metadata         = torch.tensor(REGION_META[region], dtype=torch.float32)
    ds.seq_len          = SEQ_LEN
    ds.forecast_horizon = FORECAST_HORIZON
    return ds


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 1 — DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
banner(1, "DATA PIPELINE")

FILES = [
    ("TX solar",   "pecanstreet_austin_15min_solar.csv",     "local_15min", "solar"),
    ("CA solar",   "pecanstreet_california_15min_solar.csv", "local_15min", "solar"),
    ("NY solar",   "pecanstreet_newyork_15min_solar.csv",    "local_15min", "solar"),
    ("TX weather", "nasa_power_austin_2018.csv",             "datetime",    "GHI_W_m2"),
    ("CA weather", "nasa_power_california_2018.csv",         "datetime",    "GHI_W_m2"),
    ("NY weather", "nasa_power_newyork_2019.csv",            "datetime",    "GHI_W_m2"),
    ("metadata",   "pecanstreet_metadata.csv",               None,          None),
]

pipeline_ok = True
loaded: dict[str, pl.DataFrame] = {}

for label, fname, ts_col, key_col in FILES:
    path = DATA_ROOT / fname
    try:
        df = pl.read_csv(path)
        sample = df.head(100)
        null_info = (
            f"{key_col}={sample[key_col].null_count()} nulls"
            if key_col and key_col in sample.columns else ""
        )
        ts_range = ""
        if ts_col and ts_col in df.columns:
            vals = df[ts_col].cast(pl.Utf8)
            ts_range = f"  |  {vals[0]} .. {vals[-1]}"
        print(f"  {label:<12} shape={str(df.shape):<18} cols={len(df.columns):<5}"
              f"{null_info}{ts_range}")
        loaded[label] = df
    except Exception as exc:
        print(f"  {label}: ERROR — {exc}")
        pipeline_ok = False

mark("DATA PIPELINE", pipeline_ok,
     f"all {len(FILES)} files loaded" if pipeline_ok else "file load error")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 2 — PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
banner(2, "PREPROCESSING")

preproc_ok = True
aligned_data: dict[str, pl.DataFrame] = {}

try:
    for region, solar_key, wx_key in [("tx", "TX solar", "TX weather"),
                                       ("ca", "CA solar", "CA weather")]:
        home_ids = homes_with_solar(loaded[solar_key])
        assert home_ids, f"No homes with solar data in {solar_key}"
        dataid = home_ids[0]

        aligned = align_weather_solar(loaded[solar_key], loaded[wx_key], dataid)
        assert aligned is not None, (
            f"{region.upper()} home {dataid}: alignment produced <{SEQ_LEN+FORECAST_HORIZON} rows"
        )

        # Verify no nulls in required columns
        for col in ["solar"] + WEATHER_COLS:
            n_null = aligned[col].null_count()
            assert n_null == 0, f"{col} has {n_null} nulls after alignment"

        print(f"  {region.upper()} home {dataid}:  {aligned.height} rows aligned")
        print(f"    solar    : [{aligned['solar'].min():.4f}, {aligned['solar'].max():.4f}] kWh")
        print(f"    GHI      : [{aligned['GHI_W_m2'].min():.1f}, {aligned['GHI_W_m2'].max():.1f}] W/m²")
        print(f"    Temp     : [{aligned['Temp_C'].min():.1f}, {aligned['Temp_C'].max():.1f}] °C")
        aligned_data[region] = aligned

    mark("PREPROCESSING", True, "TX and CA aligned, zero nulls in solar/GHI/DNI/temperature")
except Exception as exc:
    mark("PREPROCESSING", False, str(exc))
    traceback.print_exc()
    preproc_ok = False


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 3 — WINDOWING
# ═════════════════════════════════════════════════════════════════════════════
banner(3, "WINDOWING")

try:
    assert preproc_ok, "Skipped — preprocessing failed"
    region = next(iter(aligned_data))
    ds = make_dataset(aligned_data[region], region)
    assert len(ds) > 0, f"Dataset empty (n_rows={ds.weather.shape[0]})"

    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    weather_b, gen_b, meta_b, target_b = next(iter(loader))
    B = weather_b.shape[0]

    print(f"  Dataset length : {len(ds)} windows")
    print(f"  weather    : {tuple(weather_b.shape)}   expected ({B}, {SEQ_LEN}, {N_WEATHER})")
    print(f"  generation : {tuple(gen_b.shape)}   expected ({B}, {SEQ_LEN}, 1)")
    print(f"  metadata   : {tuple(meta_b.shape)}     expected ({B}, {N_METADATA})")
    print(f"  target     : {tuple(target_b.shape)}     expected ({B}, {FORECAST_HORIZON})")

    assert weather_b.shape == (B, SEQ_LEN, N_WEATHER),   f"weather {weather_b.shape}"
    assert gen_b.shape     == (B, SEQ_LEN, 1),            f"generation {gen_b.shape}"
    assert meta_b.shape    == (B, N_METADATA),             f"metadata {meta_b.shape}"
    assert target_b.shape  == (B, FORECAST_HORIZON),       f"target {target_b.shape}"

    mark("WINDOWING", True, f"all shapes correct, {len(ds)} windows from {region.upper()} home")
except Exception as exc:
    mark("WINDOWING", False, str(exc))
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 4 — MODEL FORWARD PASS
# ═════════════════════════════════════════════════════════════════════════════
banner(4, "MODEL FORWARD PASS")

model = None
try:
    model = build(ROOT / "configs" / "experiment.yaml")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    w  = torch.randn(BATCH, SEQ_LEN, N_WEATHER)
    g  = torch.randn(BATCH, SEQ_LEN, 1)
    m  = torch.randn(BATCH, N_METADATA)

    with torch.no_grad():
        out = model(w, g, m)

    print(f"  Parameters : {n_params:,}")
    print(f"  Inputs     : weather={tuple(w.shape)}, generation={tuple(g.shape)}, metadata={tuple(m.shape)}")
    print(f"  Output     : {tuple(out.shape)}   expected ({BATCH}, {FORECAST_HORIZON})")

    assert out.shape == (BATCH, FORECAST_HORIZON), f"shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"

    mark("MODEL FORWARD PASS", True,
         f"output={tuple(out.shape)}, {n_params:,} parameters")
except Exception as exc:
    mark("MODEL FORWARD PASS", False, str(exc))
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 5 — TRAIN/TEST SPLIT
# ═════════════════════════════════════════════════════════════════════════════
banner(5, "TRAIN/TEST SPLIT")

try:
    tx_homes = set(homes_with_solar(loaded["TX solar"]))
    ca_homes = set(homes_with_solar(loaded["CA solar"]))
    ny_homes = set(homes_with_solar(loaded["NY solar"]))

    train_homes = tx_homes | ca_homes
    overlap     = train_homes & ny_homes

    print(f"  TX train homes ({len(tx_homes)}) : {sorted(tx_homes)}")
    print(f"  CA train homes ({len(ca_homes)}) : {sorted(ca_homes)}")
    print(f"  NY test  homes ({len(ny_homes)}) : {sorted(ny_homes)}")
    print(f"  Train & Test overlap       : {overlap if overlap else 'empty (no overlap)'}")

    assert len(train_homes) > 0, "No train homes found"
    assert len(ny_homes)    > 0, "No NY test homes found"
    assert len(overlap) == 0,    f"Train/test overlap: {overlap}"

    mark("TRAIN/TEST SPLIT", True,
         f"train={len(train_homes)} (TX+CA), test={len(ny_homes)} (NY), overlap=0")
except Exception as exc:
    mark("TRAIN/TEST SPLIT", False, str(exc))
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 6 — NY TRANSFER (zero-shot forward pass)
# ═════════════════════════════════════════════════════════════════════════════
banner(6, "NY TRANSFER CHECK")

try:
    assert model is not None, "Model unavailable — check 4 failed"

    ny_home_ids = homes_with_solar(loaded["NY solar"])
    assert ny_home_ids, "No NY homes with solar data"
    ny_dataid = ny_home_ids[0]

    ny_aligned = align_weather_solar(loaded["NY solar"], loaded["NY weather"], ny_dataid)
    assert ny_aligned is not None, (
        f"NY home {ny_dataid}: alignment produced <{SEQ_LEN+FORECAST_HORIZON} rows"
    )

    ds_ny = make_dataset(ny_aligned, "ny")
    assert len(ds_ny) > 0, "NY dataset empty after alignment"

    loader_ny = DataLoader(ds_ny, batch_size=BATCH, shuffle=False)
    wx_b, gen_b, meta_b, _ = next(iter(loader_ny))

    model.eval()
    with torch.no_grad():
        preds = model(wx_b, gen_b, meta_b)

    assert not torch.isnan(preds).any(), "NY predictions contain NaN"
    expected_B = min(BATCH, len(ds_ny))
    assert preds.shape == (expected_B, FORECAST_HORIZON), f"shape {preds.shape}"

    print(f"  NY home      : {ny_dataid}  ({ny_aligned.height} aligned rows, {len(ds_ny)} windows)")
    print(f"  Output shape : {tuple(preds.shape)}")
    print(f"  Sample predictions (kWh per 15-min step, untrained model):")
    for i, row in enumerate(preds[:3].tolist()):
        steps = "  ".join(f"{v:+.4f}" for v in row)
        print(f"    window {i}: [{steps}]")

    mark("NY TRANSFER", True,
         f"zero-shot forward pass succeeded, output={tuple(preds.shape)}")
except Exception as exc:
    mark("NY TRANSFER", False, str(exc))
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
all_pass = True
for name, passed in RESULTS.items():
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if not passed:
        all_pass = False

print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print(f"{'='*60}\n")
sys.exit(0 if all_pass else 1)
