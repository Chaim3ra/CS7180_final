"""End-to-end validation: data pipeline, preprocessing, windowing, model, splits.

All data is read directly from S3 — no local files required.
Only prerequisite: AWS credentials in .env and S3_BUCKET set.

Exit 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# -- Resolve repo root and load .env ------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

env_path = ROOT / ".env"
if env_path.exists():
    for raw in env_path.read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

S3_BUCKET           = os.environ.get("S3_BUCKET", "")
S3_RAW_PREFIX       = os.environ.get("S3_RAW_PREFIX", "raw")
S3_PROCESSED_PREFIX = (
    os.environ.get("S3_PROCESSED_PREFIX")
    or os.environ.get("S3_DATA_PREFIX", "data/processed")
)

# Local fallback roots (used only when S3_BUCKET is not set)
_dr = os.environ.get("DATA_ROOT", "data/raw")
DATA_ROOT = Path(_dr) if Path(_dr).is_absolute() else ROOT / _dr
PROCESSED = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.models import build
from src.dataloader import SolarWindowDataset, read_csv, read_parquet

# -- Constants ----------------------------------------------------------------
SEQ_LEN          = 96
FORECAST_HORIZON = 4
BATCH            = 8
N_WEATHER        = 6
N_METADATA       = 6
WEATHER_COLS     = ["GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct"]
SOLAR_COL        = "solar_kwh"
METADATA_COLS    = ["lat", "lon", "tilt_deg", "azimuth_deg", "capacity_kw", "elevation_m"]

RESULTS: dict[str, bool] = {}


# -- URI helpers ---------------------------------------------------------------

def _raw_uri(filename: str) -> str:
    """Return an S3 URI or local path for a raw data file."""
    if S3_BUCKET:
        return f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/{filename}"
    return str(DATA_ROOT / filename)


def _processed_uri(filename: str) -> str:
    """Return an S3 URI or local path for a processed parquet."""
    if S3_BUCKET:
        return f"s3://{S3_BUCKET}/{S3_PROCESSED_PREFIX}/{filename}"
    return str(PROCESSED / filename)


# -- Helpers ------------------------------------------------------------------
def banner(n: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  CHECK {n} -- {title}")
    print(f"{'='*60}")


def mark(name: str, passed: bool, msg: str = "") -> None:
    RESULTS[name] = passed
    tag = "PASS" if passed else "FAIL"
    suffix = f" -- {msg}" if msg else ""
    print(f"  [{tag}] {name}{suffix}")


def make_dataset_from_parquet(df: pl.DataFrame, dataid: Optional[int] = None) -> SolarWindowDataset:
    """Build a SolarWindowDataset from a processed parquet DataFrame."""
    if dataid is not None:
        df = df.filter(pl.col("dataid") == dataid)
    meta_row = df.row(0, named=True)
    metadata = [float(meta_row[c]) for c in METADATA_COLS]

    wx  = df.select(WEATHER_COLS).to_numpy().astype(np.float32)
    gen = df.select(SOLAR_COL).to_numpy().astype(np.float32)
    n   = min(len(wx), len(gen))

    ds = SolarWindowDataset.__new__(SolarWindowDataset)
    ds.weather          = torch.tensor(wx[:n],  dtype=torch.float32)
    ds.generation       = torch.tensor(gen[:n], dtype=torch.float32)
    ds.metadata         = torch.tensor(metadata, dtype=torch.float32)
    ds.seq_len          = SEQ_LEN
    ds.forecast_horizon = FORECAST_HORIZON
    return ds


# =============================================================================
# CHECK 1 -- DATA PIPELINE (raw file reads from S3)
# =============================================================================
banner(1, "DATA PIPELINE")

if S3_BUCKET:
    print(f"  Source: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/")
else:
    print(f"  Source: {DATA_ROOT}  (S3_BUCKET not set — using local files)")

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

for label, fname, ts_col, key_col in FILES:
    uri = _raw_uri(fname)
    try:
        df = read_csv(uri)
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
    except Exception as exc:
        print(f"  {label}: ERROR -- {exc}")
        pipeline_ok = False

mark("DATA PIPELINE", pipeline_ok,
     f"all {len(FILES)} files loaded" if pipeline_ok else "file load error")


# =============================================================================
# CHECK 2 -- PREPROCESSING (validate processed parquets from S3)
# =============================================================================
banner(2, "PREPROCESSING")

if S3_BUCKET:
    print(f"  Source: s3://{S3_BUCKET}/{S3_PROCESSED_PREFIX}/")
else:
    print(f"  Source: {PROCESSED}  (S3_BUCKET not set — using local files)")

preproc_ok = True
parquets: dict[str, pl.DataFrame] = {}

try:
    PARQUET_SPECS = [
        ("TX", "train_tx.parquet", 19, []),
        ("CA", "train_ca.parquet",  1, [9836]),
        ("NY", "test_ny.parquet",  14, []),
    ]

    for region, fname, expected_n, expected_ids in PARQUET_SPECS:
        uri = _processed_uri(fname)
        df = read_parquet(uri)
        parquets[region] = df

        home_ids = sorted(df["dataid"].unique().to_list())
        n_homes  = len(home_ids)

        assert n_homes == expected_n, (
            f"{region}: expected {expected_n} homes, got {n_homes}"
        )
        if expected_ids:
            assert home_ids == expected_ids, (
                f"{region}: expected dataids {expected_ids}, got {home_ids}"
            )

        # No nulls in required columns
        for col in [SOLAR_COL] + WEATHER_COLS:
            n_null = df[col].null_count()
            assert n_null == 0, f"{region} {col}: {n_null} nulls found"

        # No all-zero solar homes (confirms quality filter ran correctly)
        for hid in home_ids:
            nonzero = df.filter((pl.col("dataid") == hid) & (pl.col(SOLAR_COL) > 0)).height
            assert nonzero > 0, (
                f"{region} dataid={hid}: all-zero solar present -- "
                "filter_solar_homes should have removed this home"
            )

        solar_min = df[SOLAR_COL].min()
        solar_max = df[SOLAR_COL].max()
        ghi_min   = df["GHI_W_m2"].min()
        ghi_max   = df["GHI_W_m2"].max()
        print(f"  {region:<4}  {fname:<26}  {df.height:>8,} rows  {n_homes} homes")
        print(f"        dataids    : {home_ids}")
        print(f"        solar_kwh  : [{solar_min:.4f}, {solar_max:.4f}]")
        print(f"        GHI_W_m2   : [{ghi_min:.1f}, {ghi_max:.1f}]")

    mark("PREPROCESSING", True,
         "all 3 parquets valid; correct home counts; no nulls; no all-zero homes")
except Exception as exc:
    mark("PREPROCESSING", False, str(exc))
    traceback.print_exc()
    preproc_ok = False


# =============================================================================
# CHECK 3 -- WINDOWING (from processed TX parquet)
# =============================================================================
banner(3, "WINDOWING")

try:
    assert preproc_ok, "Skipped -- preprocessing check failed"

    tx_df      = parquets["TX"]
    first_home = sorted(tx_df["dataid"].unique().to_list())[0]
    ds = make_dataset_from_parquet(tx_df, dataid=first_home)
    assert len(ds) > 0, f"Dataset empty (n_rows={ds.weather.shape[0]})"

    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    weather_b, gen_b, meta_b, target_b = next(iter(loader))
    B = weather_b.shape[0]

    print(f"  Source     : TX parquet, dataid={first_home}")
    print(f"  Dataset    : {len(ds)} windows")
    print(f"  weather    : {tuple(weather_b.shape)}   expected ({B}, {SEQ_LEN}, {N_WEATHER})")
    print(f"  generation : {tuple(gen_b.shape)}   expected ({B}, {SEQ_LEN}, 1)")
    print(f"  metadata   : {tuple(meta_b.shape)}     expected ({B}, {N_METADATA})")
    print(f"  target     : {tuple(target_b.shape)}     expected ({B}, {FORECAST_HORIZON})")

    assert weather_b.shape == (B, SEQ_LEN, N_WEATHER), f"weather {weather_b.shape}"
    assert gen_b.shape     == (B, SEQ_LEN, 1),          f"generation {gen_b.shape}"
    assert meta_b.shape    == (B, N_METADATA),           f"metadata {meta_b.shape}"
    assert target_b.shape  == (B, FORECAST_HORIZON),     f"target {target_b.shape}"

    mark("WINDOWING", True,
         f"all shapes correct, {len(ds)} windows from TX dataid={first_home}")
except Exception as exc:
    mark("WINDOWING", False, str(exc))
    traceback.print_exc()


# =============================================================================
# CHECK 4 -- MODEL FORWARD PASS
# =============================================================================
banner(4, "MODEL FORWARD PASS")

model = None
try:
    model = build(ROOT / "configs" / "experiment.yaml")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    w = torch.randn(BATCH, SEQ_LEN, N_WEATHER)
    g = torch.randn(BATCH, SEQ_LEN, 1)
    m = torch.randn(BATCH, N_METADATA)

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


# =============================================================================
# CHECK 5 -- TRAIN/TEST SPLIT (from processed parquets)
# =============================================================================
banner(5, "TRAIN/TEST SPLIT")

try:
    assert preproc_ok, "Skipped -- preprocessing check failed"

    tx_ids = set(parquets["TX"]["dataid"].unique().to_list())
    ca_ids = set(parquets["CA"]["dataid"].unique().to_list())
    ny_ids = set(parquets["NY"]["dataid"].unique().to_list())

    train_ids = tx_ids | ca_ids
    overlap   = train_ids & ny_ids

    print(f"  TX train ({len(tx_ids)} homes)  : {sorted(tx_ids)}")
    print(f"  CA train ({len(ca_ids)} home)    : {sorted(ca_ids)}")
    print(f"  NY test  ({len(ny_ids)} homes) : {sorted(ny_ids)}")
    print(f"  Train/test overlap : {overlap if overlap else 'empty (no overlap)'}")

    assert len(tx_ids) == 19,         f"TX: expected 19 homes, got {len(tx_ids)}"
    assert len(ca_ids) ==  1,         f"CA: expected 1 home, got {len(ca_ids)}"
    assert sorted(ca_ids) == [9836],  f"CA: expected [9836], got {sorted(ca_ids)}"
    assert len(ny_ids) == 14,         f"NY: expected 14 homes, got {len(ny_ids)}"
    assert len(overlap) == 0,         f"Train/test overlap: {overlap}"

    mark("TRAIN/TEST SPLIT", True,
         "TX=19 + CA=1 train, NY=14 test, overlap=0")
except Exception as exc:
    mark("TRAIN/TEST SPLIT", False, str(exc))
    traceback.print_exc()


# =============================================================================
# CHECK 6 -- NY TRANSFER (from processed test parquet)
# =============================================================================
banner(6, "NY TRANSFER CHECK")

try:
    assert model is not None, "Model unavailable -- check 4 failed"
    assert preproc_ok, "Skipped -- preprocessing check failed"

    ny_df       = parquets["NY"]
    ny_home_ids = sorted(ny_df["dataid"].unique().to_list())
    ny_dataid   = ny_home_ids[0]

    ds_ny = make_dataset_from_parquet(ny_df, dataid=ny_dataid)
    assert len(ds_ny) > 0, f"NY dataid={ny_dataid}: dataset empty after windowing"

    loader_ny = DataLoader(ds_ny, batch_size=BATCH, shuffle=False)
    wx_b, gen_b, meta_b, _ = next(iter(loader_ny))

    model.eval()
    with torch.no_grad():
        preds = model(wx_b, gen_b, meta_b)

    assert not torch.isnan(preds).any(), "NY predictions contain NaN"
    expected_B = min(BATCH, len(ds_ny))
    assert preds.shape == (expected_B, FORECAST_HORIZON), f"shape {preds.shape}"

    home_rows = ny_df.filter(pl.col("dataid") == ny_dataid).height
    print(f"  NY home      : {ny_dataid}  ({home_rows} rows, {len(ds_ny)} windows)")
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


# =============================================================================
# SUMMARY
# =============================================================================
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
