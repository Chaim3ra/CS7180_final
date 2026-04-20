"""Align, enrich, and save processed training/test data as Parquet files.

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
data/processed/train_tx.parquet
data/processed/train_ca.parquet
data/processed/test_ny.parquet

Caching: a region is skipped if its parquet already exists (pass --force to reprocess).
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

_dr = os.environ.get("DATA_ROOT", "data/raw")
DATA_ROOT = Path(_dr) if Path(_dr).is_absolute() else ROOT / _dr
PROCESSED = ROOT / "data" / "processed"

import polars as pl

from src.dataloader import filter_solar_homes

# -- Constants -----------------------------------------------------------------
WEATHER_COLS = ["GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct"]
MIN_ALIGNED_ROWS = 96 + 4   # seq_len + forecast_horizon

# City-level geographic defaults (Pecan Street provides no exact home coordinates)
CITY_GEO: dict[str, dict] = {
    "austin":   {"lat": 30.2672, "lon": -97.7431, "tilt_deg": 25.0, "elevation_m": 200.0},
    "san jose": {"lat": 37.3382, "lon": -121.8863, "tilt_deg": 30.0, "elevation_m": 30.0},
    "new york": {"lat": 40.7128, "lon": -74.0060,  "tilt_deg": 35.0, "elevation_m": 10.0},
}

# Fallback geo for CA cities that aren't "San Jose" in metadata
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
    "east;west":   180.0,  # split array — use annual-average equivalent
    "west;east":   180.0,
}


# -- Helpers -------------------------------------------------------------------
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


def align_home(
    solar_df: pl.DataFrame,
    weather_df: pl.DataFrame,
    dataid: int,
) -> Optional[pl.DataFrame]:
    """Left-join hourly NASA-POWER weather to 15-min solar rows for one home.

    Join key: first 13 chars of timestamp string (YYYY-MM-DD HH), which
    strips timezone offsets present in Pecan Street timestamps.

    Returns aligned DataFrame or None if insufficient rows.
    """
    home = (
        solar_df
        .filter(pl.col("dataid") == dataid)
        .select(["local_15min", "solar"])
        .with_columns(
            pl.col("local_15min").str.slice(0, 13).alias("_ts_hour")
        )
    )
    if home.height == 0:
        return None

    wx = (
        weather_df
        .with_columns(pl.col("datetime").str.slice(0, 13).alias("_ts_hour"))
        .select(["_ts_hour"] + WEATHER_COLS)
    )

    joined = (
        home
        .join(wx, on="_ts_hour", how="left")
        .drop("_ts_hour")
    )
    # Keep only rows where both solar and all weather cols are present
    joined = joined.filter(pl.col("solar").is_not_null())
    for col in WEATHER_COLS:
        joined = joined.filter(pl.col(col).is_not_null())

    return joined if joined.height >= MIN_ALIGNED_ROWS else None


def process_region(
    region: str,
    solar_path: Path,
    weather_paths: list[Path],
    meta_df: pl.DataFrame,
    out_path: Path,
    force: bool = False,
) -> Optional[pl.DataFrame]:
    """Process one region and save to parquet.  Returns the DataFrame."""
    if out_path.exists() and not force:
        print(f"  [CACHED]  {out_path.name} already exists — skipping (use --force to reprocess)")
        return pl.read_parquet(out_path)

    print(f"\n  Loading {solar_path.name} ...")
    solar_df = pl.read_csv(solar_path, columns=["dataid", "local_15min", "solar"])
    wx_parts = [pl.read_csv(p) for p in weather_paths]
    weather_df = pl.concat(wx_parts, how="diagonal") if len(wx_parts) > 1 else wx_parts[0]
    print(f"  Weather : {[p.name for p in weather_paths]}  ({weather_df.height} rows)")
    home_ids = solar_df["dataid"].unique().sort().to_list()
    print(f"  Found {len(home_ids)} homes in CSV")

    chunks: list[pl.DataFrame] = []
    skipped_insufficient = []

    for dataid in home_ids:
        aligned = align_home(solar_df, weather_df, dataid)
        if aligned is None:
            skipped_insufficient.append(dataid)
            continue

        # -- Metadata lookup --------------------------------------------------
        home_meta = meta_df.filter(pl.col("dataid") == dataid)
        if home_meta.height > 0:
            row      = home_meta.row(0, named=True)
            city     = row.get("city") or ""
            geo      = get_city_geo(city, region)
            azimuth  = direction_to_azimuth(row.get("pv_panel_direction"))
            capacity = float(row.get("total_amount_of_pv") or 0.0) or 5.0  # fallback 5 kW
        else:
            geo      = REGION_GEO_FALLBACK[region]
            azimuth  = 180.0
            capacity = 5.0

        # -- Unit conversion: Pecan Street kW -> kWh per 15-min interval -----
        # Pecan Street 15-min data is average power in kW over the interval.
        # Energy = power * time = kW * (15/60) h = kW * 0.25 h -> kWh
        aligned = aligned.with_columns(
            (pl.col("solar") * 0.25).alias("solar_kwh")
        ).drop("solar")

        # -- Add metadata columns ---------------------------------------------
        aligned = aligned.with_columns([
            pl.lit(dataid).cast(pl.Int64).alias("dataid"),
            pl.lit(geo["lat"]).cast(pl.Float32).alias("lat"),
            pl.lit(geo["lon"]).cast(pl.Float32).alias("lon"),
            pl.lit(geo["tilt_deg"]).cast(pl.Float32).alias("tilt_deg"),
            pl.lit(azimuth).cast(pl.Float32).alias("azimuth_deg"),
            pl.lit(capacity).cast(pl.Float32).alias("capacity_kw"),
            pl.lit(geo["elevation_m"]).cast(pl.Float32).alias("elevation_m"),
        ])

        # Cast weather to Float32
        for col in WEATHER_COLS:
            aligned = aligned.with_columns(pl.col(col).cast(pl.Float32))

        chunks.append(aligned)

    if skipped_insufficient:
        print(f"  [SKIPPED] {len(skipped_insufficient)} homes with <{MIN_ALIGNED_ROWS} aligned rows: "
              f"{skipped_insufficient}")

    if not chunks:
        print(f"  [ERROR] No aligned data for region {region.upper()}")
        return None

    if not chunks:
        print(f"  [ERROR] No aligned data for region {region.upper()} — check weather file coverage")
        return None

    df = pl.concat(chunks, how="diagonal")
    print(f"\n  Combined {region.upper()} dataset: {df.height:,} rows, {df['dataid'].n_unique()} homes")

    # -- Apply solar quality filter -------------------------------------------
    print(f"\n  Filtering homes with <5% non-zero solar...")
    valid_ids = filter_solar_homes(df, solar_col="solar_kwh", min_nonzero_frac=0.05)
    n_filtered = df["dataid"].n_unique() - len(valid_ids)
    df = df.filter(pl.col("dataid").is_in(valid_ids))
    print(f"  Kept {len(valid_ids)} homes, removed {n_filtered}")

    if df.height == 0:
        print(f"  [ERROR] All homes filtered out for {region.upper()} — no data to save")
        return None

    # -- Summary stats --------------------------------------------------------
    print(f"\n  {region.upper()} final: {df.height:,} rows, {df['dataid'].n_unique()} homes")
    print(f"    solar_kwh : [{df['solar_kwh'].min():.4f}, {df['solar_kwh'].max():.4f}]  "
          f"mean={df['solar_kwh'].mean():.4f}")
    print(f"    GHI_W_m2  : [{df['GHI_W_m2'].min():.1f}, {df['GHI_W_m2'].max():.1f}]")
    print(f"    homes     : {sorted(df['dataid'].unique().to_list())}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"  Saved -> {out_path.relative_to(ROOT)}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main(force: bool = False) -> None:
    print("=" * 60)
    print("  PREPROCESSING")
    print("=" * 60)
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print(f"  PROCESSED : {PROCESSED}")

    meta_df = pl.read_csv(DATA_ROOT / "pecanstreet_metadata.csv")
    print(f"  Metadata  : {meta_df.height} homes loaded")

    # CA home 9836 spans 2014-2015; load all available CA weather years
    ca_wx = sorted(DATA_ROOT.glob("nasa_power_california_*.csv"))

    REGIONS = [
        ("tx", "pecanstreet_austin_15min_solar.csv",    ["nasa_power_austin_2018.csv"],  "train_tx.parquet"),
        ("ca", "pecanstreet_california_15min_solar.csv", [p.name for p in ca_wx],        "train_ca.parquet"),
        ("ny", "pecanstreet_newyork_15min_solar.csv",   ["nasa_power_newyork_2019.csv"], "test_ny.parquet"),
    ]

    results: dict[str, int] = {}
    for region, solar_f, wx_files, out_f in REGIONS:
        print(f"\n{'-'*60}")
        print(f"  Region: {region.upper()}")
        print(f"{'-'*60}")
        df = process_region(
            region=region,
            solar_path=DATA_ROOT / solar_f,
            weather_paths=[DATA_ROOT / f for f in wx_files],
            meta_df=meta_df,
            out_path=PROCESSED / out_f,
            force=force,
        )
        results[region] = df["dataid"].n_unique() if df is not None else 0

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    for region, n_homes in results.items():
        tag = "train" if region != "ny" else "test"
        print(f"  {region.upper()} ({tag}): {n_homes} homes")
    train_homes = results.get("tx", 0) + results.get("ca", 0)
    test_homes  = results.get("ny", 0)
    print(f"\n  Train total : {train_homes} homes (TX + CA)")
    print(f"  Test total  : {test_homes} homes (NY, zero-shot)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess solar data to parquet.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if output parquet files already exist.")
    args = parser.parse_args()
    main(force=args.force)
