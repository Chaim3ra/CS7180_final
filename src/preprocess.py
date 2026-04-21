"""Align, enrich, and save processed training/test data as Parquet files.

Raw CSVs are read from local disk (DATA_ROOT) if present, or streamed
directly from S3 when S3_BUCKET is set — no local data required.
Processed parquets are written locally (if PROCESSED dir is writable) and
uploaded to S3 when S3_BUCKET is set.

Output schema per row
---------------------
dataid          Int64
local_15min     Utf8
solar_kwh       Float32  -- Pecan Street kW * 0.25 h  = kWh per 15-min interval
GHI_W_m2        Float32
DNI_W_m2        Float32
DHI_W_m2        Float32
Temp_C          Float32
WindSpeed_m_s   Float32
RelHumidity_pct Float32
lat             Float32
lon             Float32
tilt_deg        Float32
azimuth_deg     Float32
capacity_kw     Float32
elevation_m     Float32

Outputs
-------
data/processed/train_tx.parquet  (local, if writable)
data/processed/train_ca.parquet
data/processed/test_ny.parquet
s3://<S3_BUCKET>/<S3_PROCESSED_PREFIX>/train_tx.parquet  (if S3_BUCKET set)
s3://<S3_BUCKET>/<S3_PROCESSED_PREFIX>/train_ca.parquet
s3://<S3_BUCKET>/<S3_PROCESSED_PREFIX>/test_ny.parquet

Caching
-------
A region is skipped if its parquet exists locally OR in S3 (--force overrides).
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import os
import sys
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for raw in env_path.read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_dr = os.environ.get("DATA_ROOT", "data/raw")
DATA_ROOT = Path(_dr) if Path(_dr).is_absolute() else ROOT / _dr
PROCESSED = ROOT / "data" / "processed"

# S3 configuration (all optional — pipeline works locally without them)
S3_BUCKET           = os.environ.get("S3_BUCKET", "")
S3_RAW_PREFIX       = os.environ.get("S3_RAW_PREFIX", "raw")
S3_PROCESSED_PREFIX = (
    os.environ.get("S3_PROCESSED_PREFIX")
    or os.environ.get("S3_DATA_PREFIX", "data/processed")
)

import polars as pl

from src.dataloader import filter_solar_homes, read_csv, write_parquet

# -- Constants -----------------------------------------------------------------
WEATHER_COLS = ["GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct"]
MIN_ALIGNED_ROWS = 96 + 4   # seq_len + forecast_horizon

CITY_GEO: dict[str, dict] = {
    "austin":   {"lat": 30.2672, "lon": -97.7431, "tilt_deg": 25.0, "elevation_m": 200.0},
    "san jose": {"lat": 37.3382, "lon": -121.8863, "tilt_deg": 30.0, "elevation_m": 30.0},
    "new york": {"lat": 40.7128, "lon": -74.0060,  "tilt_deg": 35.0, "elevation_m": 10.0},
}

REGION_GEO_FALLBACK: dict[str, dict] = {
    "tx": CITY_GEO["austin"],
    "ca": CITY_GEO["san jose"],
    "ny": CITY_GEO["new york"],
}

DIRECTION_AZIMUTH: dict[str, float] = {
    "south":       180.0,
    "north":       0.0,
    "east":        90.0,
    "west":        270.0,
    "south;east":  135.0,
    "south;west":  225.0,
    "east;west":   180.0,
    "west;east":   180.0,
}


# -- S3 helpers ----------------------------------------------------------------

def _s3_client():
    import boto3
    return boto3.client("s3")


def _s3_exists(bucket: str, key: str) -> bool:
    """Return True if the S3 object exists."""
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _resolve_raw(filename: str) -> str:
    """Return a local path string or S3 URI for a raw data file.

    Priority: local file (DATA_ROOT) > S3 (S3_BUCKET/S3_RAW_PREFIX).
    """
    local = DATA_ROOT / filename
    if local.exists():
        return str(local)
    if S3_BUCKET:
        return f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/{filename}"
    raise FileNotFoundError(
        f"Raw file not found locally: {local}\n"
        f"Set S3_BUCKET in .env to stream from S3, or run: dvc pull"
    )


def _glob_raw(pattern: str) -> list[str]:
    """Return sorted paths/URIs for raw files matching a filename glob.

    Checks local DATA_ROOT first; falls back to listing the S3 raw prefix.
    """
    local_matches = sorted(DATA_ROOT.glob(pattern))
    if local_matches:
        return [str(p) for p in local_matches]
    if S3_BUCKET:
        prefix = f"{S3_RAW_PREFIX}/"
        paginator = _s3_client().get_paginator("list_objects_v2")
        uris: list[str] = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key  = obj["Key"]
                fname = key.split("/")[-1]
                if fnmatch.fnmatch(fname, pattern):
                    uris.append(f"s3://{S3_BUCKET}/{key}")
        return sorted(uris)
    raise FileNotFoundError(
        f"No raw files matching '{pattern}' found locally or in S3."
    )


def _resolve_processed(filename: str) -> str:
    """Return a local path string or S3 URI for a processed parquet."""
    local = PROCESSED / filename
    if local.exists():
        return str(local)
    if S3_BUCKET:
        return f"s3://{S3_BUCKET}/{S3_PROCESSED_PREFIX}/{filename}"
    return str(local)   # return local path even if absent (write will create it)


# -- Domain helpers ------------------------------------------------------------

def direction_to_azimuth(direction: Optional[str]) -> float:
    if not direction:
        return 180.0
    return DIRECTION_AZIMUTH.get(direction.lower().strip(), 180.0)


def get_city_geo(city_str: Optional[str], region: str) -> dict:
    if city_str:
        for key, geo in CITY_GEO.items():
            if key in city_str.lower():
                return geo
    return REGION_GEO_FALLBACK[region]


# -- Alignment -----------------------------------------------------------------

def align_home(
    solar_df: pl.DataFrame,
    weather_df: pl.DataFrame,
    dataid: int,
) -> Optional[pl.DataFrame]:
    """Left-join hourly NASA-POWER weather to 15-min solar rows for one home."""
    home = (
        solar_df
        .filter(pl.col("dataid") == dataid)
        .select(["local_15min", "solar"])
        .with_columns(pl.col("local_15min").str.slice(0, 13).alias("_ts_hour"))
    )
    if home.height == 0:
        return None

    wx = (
        weather_df
        .with_columns(pl.col("datetime").str.slice(0, 13).alias("_ts_hour"))
        .select(["_ts_hour"] + WEATHER_COLS)
    )

    joined = home.join(wx, on="_ts_hour", how="left").drop("_ts_hour")
    joined = joined.filter(pl.col("solar").is_not_null())
    for col in WEATHER_COLS:
        joined = joined.filter(pl.col(col).is_not_null())

    return joined if joined.height >= MIN_ALIGNED_ROWS else None


# -- Region processing ---------------------------------------------------------

def process_region(
    region: str,
    solar_path: Union[str, Path],
    weather_paths: list[Union[str, Path]],
    meta_df: pl.DataFrame,
    out_filename: str,
    force: bool = False,
) -> Optional[pl.DataFrame]:
    """Process one region and save to parquet locally and/or S3.

    Args:
        region:        Region key ('tx', 'ca', 'ny').
        solar_path:    Local path or S3 URI to the 15-min solar CSV.
        weather_paths: List of local paths or S3 URIs to NASA POWER CSVs.
        meta_df:       Pecan Street metadata DataFrame (already loaded).
        out_filename:  Output parquet filename (e.g. 'train_tx.parquet').
        force:         Re-process even if output already exists.

    Returns:
        Processed DataFrame, or None on failure.
    """
    local_out  = PROCESSED / out_filename
    s3_key_out = f"{S3_PROCESSED_PREFIX}/{out_filename}"

    # -- Cache check -----------------------------------------------------------
    if not force:
        if local_out.exists():
            print(f"  [CACHED local]  {out_filename}")
            return pl.read_parquet(local_out)
        if S3_BUCKET and _s3_exists(S3_BUCKET, s3_key_out):
            s3_uri = f"s3://{S3_BUCKET}/{s3_key_out}"
            print(f"  [CACHED S3]     {s3_uri}")
            from src.dataloader import read_parquet
            return read_parquet(s3_uri)

    # -- Load raw data ---------------------------------------------------------
    print(f"\n  Loading solar : {solar_path}")
    solar_df = read_csv(solar_path, columns=["dataid", "local_15min", "solar"])

    wx_parts  = [read_csv(p) for p in weather_paths]
    weather_df = pl.concat(wx_parts, how="diagonal") if len(wx_parts) > 1 else wx_parts[0]
    print(f"  Weather files : {[str(p).split('/')[-1] for p in weather_paths]}")

    home_ids = solar_df["dataid"].unique().sort().to_list()
    print(f"  Found {len(home_ids)} homes in CSV")

    chunks: list[pl.DataFrame] = []
    skipped_insufficient: list[int] = []

    for dataid in home_ids:
        aligned = align_home(solar_df, weather_df, dataid)
        if aligned is None:
            skipped_insufficient.append(dataid)
            continue

        home_meta = meta_df.filter(pl.col("dataid") == dataid)
        if home_meta.height > 0:
            row      = home_meta.row(0, named=True)
            city     = row.get("city") or ""
            geo      = get_city_geo(city, region)
            azimuth  = direction_to_azimuth(row.get("pv_panel_direction"))
            capacity = float(row.get("total_amount_of_pv") or 0.0) or 5.0
        else:
            geo      = REGION_GEO_FALLBACK[region]
            azimuth  = 180.0
            capacity = 5.0

        # kW -> kWh per 15-min interval
        aligned = aligned.with_columns(
            (pl.col("solar") * 0.25).alias("solar_kwh")
        ).drop("solar")

        aligned = aligned.with_columns([
            pl.lit(dataid).cast(pl.Int64).alias("dataid"),
            pl.lit(geo["lat"]).cast(pl.Float32).alias("lat"),
            pl.lit(geo["lon"]).cast(pl.Float32).alias("lon"),
            pl.lit(geo["tilt_deg"]).cast(pl.Float32).alias("tilt_deg"),
            pl.lit(azimuth).cast(pl.Float32).alias("azimuth_deg"),
            pl.lit(capacity).cast(pl.Float32).alias("capacity_kw"),
            pl.lit(geo["elevation_m"]).cast(pl.Float32).alias("elevation_m"),
        ])
        for col in WEATHER_COLS:
            aligned = aligned.with_columns(pl.col(col).cast(pl.Float32))

        chunks.append(aligned)

    if skipped_insufficient:
        print(f"  [SKIPPED] {len(skipped_insufficient)} homes with <{MIN_ALIGNED_ROWS} rows")

    if not chunks:
        print(f"  [ERROR] No aligned data for {region.upper()}")
        return None

    df = pl.concat(chunks, how="diagonal")
    print(f"\n  Combined {region.upper()}: {df.height:,} rows, {df['dataid'].n_unique()} homes")

    # -- Quality filter --------------------------------------------------------
    print("  Filtering homes with <5% non-zero solar...")
    valid_ids  = filter_solar_homes(df, solar_col="solar_kwh", min_nonzero_frac=0.05)
    n_filtered = df["dataid"].n_unique() - len(valid_ids)
    df = df.filter(pl.col("dataid").is_in(valid_ids))
    print(f"  Kept {len(valid_ids)} homes, removed {n_filtered}")

    if df.height == 0:
        print(f"  [ERROR] All homes filtered for {region.upper()}")
        return None

    print(f"\n  {region.upper()} final: {df.height:,} rows, {df['dataid'].n_unique()} homes")
    print(f"    solar_kwh : [{df['solar_kwh'].min():.4f}, {df['solar_kwh'].max():.4f}]  "
          f"mean={df['solar_kwh'].mean():.4f}")
    print(f"    GHI_W_m2  : [{df['GHI_W_m2'].min():.1f}, {df['GHI_W_m2'].max():.1f}]")
    print(f"    homes     : {sorted(df['dataid'].unique().to_list())}")

    # -- Write locally ---------------------------------------------------------
    local_out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(local_out)
    print(f"  Saved local -> {local_out.relative_to(ROOT)}")

    # -- Upload to S3 ----------------------------------------------------------
    if S3_BUCKET:
        s3_uri = f"s3://{S3_BUCKET}/{s3_key_out}"
        write_parquet(df, s3_uri)
        print(f"  Uploaded S3 -> {s3_uri}")

    return df


# -- Synthetic CA augmentation -------------------------------------------------

def _augment_ca_synthetic(real_ca_df: pl.DataFrame) -> pl.DataFrame:
    """Concatenate synthetic CA homes with real CA homes and overwrite outputs."""
    from src.dataloader import read_parquet

    syn_local   = PROCESSED / "train_ca_synthetic.parquet"
    syn_s3_key  = f"{S3_PROCESSED_PREFIX}/train_ca_synthetic.parquet"

    print(f"\n{'-'*60}")
    print("  Augmenting CA with synthetic homes")
    print(f"{'-'*60}")

    if syn_local.exists():
        print(f"  Loading synthetic CA : {syn_local.relative_to(ROOT)}")
        syn_df = pl.read_parquet(syn_local)
    elif S3_BUCKET and _s3_exists(S3_BUCKET, syn_s3_key):
        syn_uri = f"s3://{S3_BUCKET}/{syn_s3_key}"
        print(f"  Loading synthetic CA : {syn_uri}")
        syn_df = read_parquet(syn_uri)
    else:
        raise FileNotFoundError(
            f"Synthetic CA parquet not found at {syn_local} or "
            f"s3://{S3_BUCKET}/{syn_s3_key}. Run src/synthetic.py first."
        )

    n_real = real_ca_df["dataid"].n_unique()
    n_syn  = syn_df["dataid"].n_unique()
    print(f"  Real CA homes      : {n_real}")
    print(f"  Synthetic CA homes : {n_syn}")

    combined = pl.concat([real_ca_df, syn_df], how="diagonal")
    print(f"  Combined CA homes  : {combined['dataid'].n_unique()} ({combined.height:,} rows)")

    local_out = PROCESSED / "train_ca.parquet"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(local_out)
    print(f"  Saved local -> {local_out.relative_to(ROOT)}")

    if S3_BUCKET:
        s3_uri = f"s3://{S3_BUCKET}/{S3_PROCESSED_PREFIX}/train_ca.parquet"
        write_parquet(combined, s3_uri)
        print(f"  Uploaded S3 -> {s3_uri}")

    return combined


# -- Entry point ---------------------------------------------------------------

def main(force: bool = False) -> None:
    print("=" * 60)
    print("  PREPROCESSING")
    print("=" * 60)
    print(f"  DATA_ROOT           : {DATA_ROOT}")
    print(f"  PROCESSED           : {PROCESSED}")
    print(f"  S3_BUCKET           : {S3_BUCKET or '(not set — local only)'}")
    if S3_BUCKET:
        print(f"  S3_RAW_PREFIX       : {S3_RAW_PREFIX}")
        print(f"  S3_PROCESSED_PREFIX : {S3_PROCESSED_PREFIX}")

    meta_path = _resolve_raw("pecanstreet_metadata.csv")
    print(f"\n  Loading metadata from: {meta_path}")
    meta_df = read_csv(meta_path)
    print(f"  Metadata: {meta_df.height} homes loaded")

    ca_wx_paths = _glob_raw("nasa_power_california_*.csv")
    print(f"  CA weather files: {[str(p).split('/')[-1] for p in ca_wx_paths]}")

    REGIONS = [
        ("tx", "pecanstreet_austin_15min_solar.csv",
                ["nasa_power_austin_2018.csv"],
                "train_tx.parquet"),
        ("ca", "pecanstreet_california_15min_solar.csv",
                [str(p).split("/")[-1] if not str(p).startswith("s3://")
                 else str(p).split("/")[-1] for p in ca_wx_paths],
                "train_ca.parquet"),
        ("ny", "pecanstreet_newyork_15min_solar.csv",
                ["nasa_power_newyork_2019.csv"],
                "test_ny.parquet"),
    ]

    results: dict[str, int] = {}
    ca_df: Optional[pl.DataFrame] = None
    for region, solar_f, wx_files, out_f in REGIONS:
        print(f"\n{'-'*60}")
        print(f"  Region: {region.upper()}")
        print(f"{'-'*60}")

        solar_path   = _resolve_raw(solar_f)
        weather_paths = [_resolve_raw(f) for f in wx_files]

        df = process_region(
            region=region,
            solar_path=solar_path,
            weather_paths=weather_paths,
            meta_df=meta_df,
            out_filename=out_f,
            force=force,
        )
        results[region] = df["dataid"].n_unique() if df is not None else 0
        if region == "ca":
            ca_df = df

    # Augment CA with synthetic homes
    if ca_df is not None:
        combined_ca = _augment_ca_synthetic(ca_df)
        results["ca"] = combined_ca["dataid"].n_unique()

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    for region, n_homes in results.items():
        tag = "train" if region != "ny" else "test"
        print(f"  {region.upper()} ({tag}): {n_homes} homes")
    print(f"\n  Train total : {results.get('tx', 0) + results.get('ca', 0)} homes (TX + CA)")
    print(f"  Test total  : {results.get('ny', 0)} homes (NY, zero-shot)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess solar data to parquet.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if output parquet files already exist.")
    args = parser.parse_args()
    main(force=args.force)
