#!/usr/bin/env python3
"""
src/synthetic.py

Generates 18 synthetic San Diego solar homes using:
  - Pecan Street Civita host homes (building type stratification)
  - Tracking the Sun San Diego RES_SF panel parameter distributions
  - NASA POWER 2014-2015 hourly weather resampled to 15-min
  - pvlib PVWatts ModelChain for AC generation
  - Gaussian noise calibrated to real home 9836 per-hour variance

Output: data/processed/train_ca_synthetic.parquet
        s3://cs7180-final-project/data/processed/train_ca_synthetic.parquet
        results/v2/synthetic_vs_real_daily.png
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain

warnings.filterwarnings("ignore")

SEED = 42
RNG = np.random.default_rng(SEED)

CIVITA_LAT = 32.7849
CIVITA_LON = -117.1539
CIVITA_ELEV = 10.0  # metres above sea level

DATE_START = "2014-07-08"
DATE_END = "2015-06-30"

BT_TARGETS = {
    "Apartment": 9,
    "Town Home": 7,
    "Single-Family Home 001 (Master)": 2,
}

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS_V2 = ROOT / "results" / "v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pr(s: str) -> None:
    sys.stdout.buffer.write((s + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


# ---------------------------------------------------------------------------
# 1. Host home selection
# ---------------------------------------------------------------------------

def load_host_homes() -> pl.DataFrame:
    """Stratified sample from 57 Civita Pecan Street homes by building_type."""
    meta = pl.read_csv(DATA_RAW / "sandiego_homes_metadata.csv", infer_schema_length=0)
    selected = []
    for bt, n in BT_TARGETS.items():
        pool = meta.filter(pl.col("building_type") == bt)
        n_pick = min(n, len(pool))
        idx = RNG.choice(len(pool), size=n_pick, replace=False)
        selected.append(pool[idx.tolist()])
    return pl.concat(selected)


# ---------------------------------------------------------------------------
# 2. Panel parameter sampling
# ---------------------------------------------------------------------------

def sample_panel_params(tts_res: pl.DataFrame, host_homes: pl.DataFrame) -> list[dict]:
    """Sample system size, tilt, azimuth from TTS RES_SF p10-p90 per home."""
    for col in ["PV_system_size_DC", "tilt_1", "azimuth_1"]:
        tts_res = tts_res.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    def bounded(col: str, lo: float, hi: float) -> np.ndarray:
        arr = tts_res[col].drop_nulls().to_numpy()
        return arr[(arr >= lo) & (arr <= hi)]

    size_all = bounded("PV_system_size_DC", 0.5, 50.0)
    tilt_all = bounded("tilt_1", 0.0, 60.0)
    az_all   = bounded("azimuth_1", 0.0, 360.0)

    def in_pct(arr: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
        return arr[(arr >= np.percentile(arr, lo_pct)) & (arr <= np.percentile(arr, hi_pct))]

    size_apt  = in_pct(size_all, 10, 50)   # apartments  → lower half
    size_town = in_pct(size_all, 10, 90)   # town homes  → full p10-p90
    size_sfh  = in_pct(size_all, 50, 90)   # SFH         → upper half
    tilt_pool = in_pct(tilt_all, 10, 90)
    az_pool   = in_pct(az_all,   10, 90)

    size_pools = {
        "Apartment": size_apt,
        "Town Home": size_town,
        "Single-Family Home 001 (Master)": size_sfh,
    }

    params = []
    for i, row in enumerate(host_homes.iter_rows(named=True)):
        bt = row["building_type"]
        pool = size_pools.get(bt, size_town)
        params.append({
            "dataid":         f"syn_sd_{i+1:03d}",
            "building_type":  bt,
            "lat":            round(CIVITA_LAT + RNG.uniform(-0.005, 0.005), 6),
            "lon":            round(CIVITA_LON + RNG.uniform(-0.005, 0.005), 6),
            "system_size_kw": round(float(RNG.choice(pool)), 3),
            "tilt_deg":       round(float(RNG.choice(tilt_pool)), 1),
            "azimuth_deg":    round(float(RNG.choice(az_pool)), 1),
            "module_type":    "Mono-c-Si" if RNG.random() < 0.807 else "Multi-c-Si",
        })
    return params


# ---------------------------------------------------------------------------
# 3. Weather loading
# ---------------------------------------------------------------------------

def load_weather() -> pd.DataFrame:
    """
    Load NASA POWER CA 2014+2015 hourly, concat, resample to 15-min (linear),
    filter to DATE_START → DATE_END, return UTC-indexed DataFrame.
    """
    dfs = []
    for yr in [2014, 2015]:
        df = pd.read_csv(
            DATA_RAW / f"nasa_power_california_{yr}.csv",
            comment="#",
            parse_dates=["datetime"],
        )
        dfs.append(df)
    wx = pd.concat(dfs, ignore_index=True).set_index("datetime")
    wx.index = pd.to_datetime(wx.index).tz_localize("UTC")
    wx = wx.rename(columns={
        "GHI_W_m2":      "ghi",
        "DNI_W_m2":      "dni",
        "DHI_W_m2":      "dhi",
        "Temp_C":        "temp_air",
        "WindSpeed_m_s": "wind_speed",
    })
    wx15 = wx.resample("15min").interpolate(method="linear")
    for c in ["ghi", "dni", "dhi"]:
        wx15[c] = wx15[c].clip(lower=0.0)
    start = pd.Timestamp(DATE_START, tz="UTC")
    end   = pd.Timestamp(DATE_END + " 23:45:00", tz="UTC")
    return wx15.loc[start:end]


# ---------------------------------------------------------------------------
# 4. pvlib simulation (PVWatts)
# ---------------------------------------------------------------------------

def simulate_home(params: dict, weather_15: pd.DataFrame) -> pd.Series:
    """
    Run pvlib PVWatts ModelChain for one home.
    Returns AC kWh per 15-min as a UTC-indexed Series.
    """
    loc = Location(
        latitude=params["lat"],
        longitude=params["lon"],
        tz="America/Los_Angeles",
        altitude=CIVITA_ELEV,
    )
    kw = params["system_size_kw"]
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    system = PVSystem(
        surface_tilt=params["tilt_deg"],
        surface_azimuth=params["azimuth_deg"],
        module_parameters={"pdc0": kw * 1000.0, "gamma_pdc": -0.004},
        inverter_parameters={"pdc0": kw * 1000.0, "eta_inv_nom": 0.96},
        temperature_model_parameters=temp_params,
    )
    mc = ModelChain.with_pvwatts(system, loc)
    mc.run_model(weather_15[["ghi", "dni", "dhi", "temp_air", "wind_speed"]])
    ac_w = pd.Series(mc.results.ac, index=weather_15.index).clip(lower=0.0)
    return ac_w * 0.25 / 1000.0  # W → kWh per 15-min


# ---------------------------------------------------------------------------
# 5. Noise calibration
# ---------------------------------------------------------------------------

def load_real_9836_15min() -> pd.Series:
    """Load home 9836 1-min data and resample to 15-min mean."""
    raw = (
        pl.read_csv(DATA_RAW / "pecanstreet_california_1min_solar.csv", infer_schema_length=0)
        .filter(pl.col("dataid") == "9836")
        .select(["localminute", "solar"])
        .to_pandas()
    )
    raw["localminute"] = pd.to_datetime(raw["localminute"], utc=True)
    raw = raw.set_index("localminute")
    raw["solar"] = pd.to_numeric(raw["solar"], errors="coerce").fillna(0.0)
    return raw["solar"].resample("1min").mean().resample("15min").mean()


def compute_noise_profile(real_15: pd.Series) -> np.ndarray:
    """Per-hour-of-day std of real 15-min generation (shape: 24,)."""
    df = pd.DataFrame({"solar": real_15, "hour": real_15.index.hour})
    return df.groupby("hour")["solar"].std().fillna(0.0).values


def add_noise(kwh: pd.Series, noise_by_hour: np.ndarray) -> pd.Series:
    """Add zero-mean Gaussian noise matching per-hour variance; clip to ≥0."""
    std = np.array([noise_by_hour[h] for h in kwh.index.hour])
    return pd.Series(np.clip(kwh.values + RNG.normal(0.0, std), 0.0, None), index=kwh.index)


# ---------------------------------------------------------------------------
# 6. Output formatting
# ---------------------------------------------------------------------------

def build_home_df(
    dataid: str,
    kwh: pd.Series,
    schema_cols: list[str],
) -> pl.DataFrame:
    """
    Build a Polars DataFrame matching the Pecan Street CA solar schema.
    Timestamps formatted as 'YYYY-MM-DD HH:MM:SS-05' (fixed -05 offset,
    matching home 9836 convention from Pecan Street Dataport).
    """
    ts_utcm5 = kwh.index.tz_convert("Etc/GMT+5")  # POSIX Etc/GMT+5 = UTC-5
    timestamps = [t.strftime("%Y-%m-%d %H:%M:%S-05") for t in ts_utcm5]

    rows: dict = {
        "dataid":      [dataid] * len(kwh),
        "localminute": timestamps,
        "solar":       [round(float(v), 6) for v in kwh.values],
    }
    for col in schema_cols:
        if col not in rows:
            rows[col] = [None] * len(kwh)

    return pl.DataFrame({c: rows[c] for c in schema_cols})


# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------

def save_daily_curve_plot(
    params_list: list[dict],
    home_kwh: dict[str, pd.Series],
    real_15: pd.Series,
) -> None:
    """Save mean daily generation curve (synthetic vs real 9836)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        RESULTS_V2.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))

        # Real home 9836 — align index to HH:MM for averaging
        real_by_tod = real_15.groupby(real_15.index.time).mean()
        times_real = [t.hour + t.minute / 60 for t in real_by_tod.index]
        ax.plot(times_real, real_by_tod.values, color="black", lw=2.5,
                label="Home 9836 (real)", zorder=5)

        # Each synthetic home
        for p in params_list:
            s = home_kwh[p["dataid"]]
            by_tod = s.groupby(s.index.hour + s.index.minute / 60).mean()
            ax.plot(by_tod.index, by_tod.values, alpha=0.4, lw=0.8,
                    label=p["dataid"])

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean kWh per 15-min")
        ax.set_title("Synthetic San Diego homes — mean daily curve vs real home 9836")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=6, ncol=4, loc="upper left")
        plt.tight_layout()
        out = RESULTS_V2 / "synthetic_vs_real_daily.png"
        plt.savefig(out, dpi=150)
        plt.close()
        pr(f"Plot saved: {out}")
    except Exception as e:
        pr(f"Plot skipped: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    pr("=== Synthetic CA Solar Pipeline ===\n")
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # --- host homes ---
    pr("Loading host homes (stratified sample from Civita Pecan Street)...")
    host_homes = load_host_homes()
    bt_counts = host_homes.group_by("building_type").len().sort("building_type")
    for row in bt_counts.iter_rows(named=True):
        pr(f"  {row['building_type']}: {row['len']}")

    # --- panel parameters ---
    pr("\nSampling panel parameters from Tracking the Sun RES_SF...")
    tts = pl.read_csv(DATA_RAW / "tracking_the_sun_sandiego.csv", infer_schema_length=0)
    tts_res = tts.filter(pl.col("customer_segment") == "RES_SF")
    pr(f"  RES_SF pool: {len(tts_res):,} installations")
    params_list = sample_panel_params(tts_res, host_homes)

    # --- weather ---
    pr("\nLoading NASA POWER CA weather (2014+2015, 15-min interpolation)...")
    weather_15 = load_weather()
    pr(f"  Steps: {len(weather_15):,}  "
       f"({weather_15.index[0]} → {weather_15.index[-1]})")

    # --- noise profile ---
    pr("\nComputing noise profile from home 9836...")
    real_15 = load_real_9836_15min()
    noise_by_hour = compute_noise_profile(real_15)
    pr(f"  Noise std by hour: mean={noise_by_hour.mean():.4f}  "
       f"max={noise_by_hour.max():.4f} kWh/15-min")

    # --- schema ---
    schema_cols = pl.read_csv(
        DATA_RAW / "pecanstreet_california_1min_solar.csv",
        n_rows=1, infer_schema_length=0,
    ).columns

    # --- simulate ---
    pr("\nSimulating homes:")
    pr(f"  {'dataid':<14} {'building_type':<35} {'kW':>6} {'tilt':>6} "
       f"{'az':>6}  daily_kWh")
    pr("  " + "-" * 80)

    all_dfs: list[pl.DataFrame] = []
    home_kwh: dict[str, pd.Series] = {}
    summaries: list[dict] = []

    for p in params_list:
        kwh = simulate_home(p, weather_15)
        kwh = add_noise(kwh, noise_by_hour)
        home_kwh[p["dataid"]] = kwh

        n_days = (weather_15.index[-1] - weather_15.index[0]).days + 1
        total      = float(kwh.sum())
        mean_daily = total / n_days
        summaries.append({**p, "mean_daily_kwh": round(mean_daily, 3),
                                "total_kwh":      round(total, 1)})

        pr(f"  {p['dataid']:<14} {p['building_type']:<35} "
           f"{p['system_size_kw']:>6.2f} {p['tilt_deg']:>6.1f} "
           f"{p['azimuth_deg']:>6.1f}  {mean_daily:.3f}")

        all_dfs.append(build_home_df(p["dataid"], kwh, schema_cols))

    # --- combine ---
    combined = pl.concat(all_dfs)
    pr(f"\nCombined: {len(combined):,} rows × {len(combined.columns)} cols")

    # --- save parquet ---
    out_path = DATA_PROC / "train_ca_synthetic.parquet"
    combined.write_parquet(str(out_path))
    pr(f"Saved local:  {out_path}")

    # --- S3 upload ---
    try:
        import boto3
        bucket = os.environ.get("S3_BUCKET", "cs7180-final-project")
        key    = "data/processed/train_ca_synthetic.parquet"
        pr(f"Uploading to s3://{bucket}/{key} ...")
        boto3.client("s3").upload_file(str(out_path), bucket, key)
        pr(f"Uploaded:     s3://{bucket}/{key}")
    except Exception as exc:
        pr(f"S3 upload skipped: {exc}")

    # --- plot ---
    save_daily_curve_plot(params_list, home_kwh, real_15)

    # --- summary table ---
    pr("\n=== Per-Home Summary ===")
    pr(f"{'dataid':<14} {'building_type':<35} {'kW':>6} {'tilt':>5} "
       f"{'az':>5}  {'mean_day_kWh':>13}  {'total_kWh':>10}")
    pr("-" * 98)
    for s in summaries:
        pr(f"{s['dataid']:<14} {s['building_type']:<35} "
           f"{s['system_size_kw']:>6.2f} {s['tilt_deg']:>5.1f} "
           f"{s['azimuth_deg']:>5.1f}  {s['mean_daily_kwh']:>13.3f}  "
           f"{s['total_kwh']:>10.1f}")

    # --- schema validation ---
    pr("\n=== Schema Validation ===")
    real_cols  = set(schema_cols)
    synth_cols = set(combined.columns)
    match      = real_cols == synth_cols
    pr(f"  Column sets match: {match}")
    if not match:
        pr(f"  Missing : {real_cols  - synth_cols}")
        pr(f"  Extra   : {synth_cols - real_cols}")
    pr(f"  Rows per home: {len(combined) // len(params_list):,}")
    pr(f"  Date range (first home): "
       f"{combined.filter(pl.col('dataid') == params_list[0]['dataid'])['localminute'].min()} → "
       f"{combined.filter(pl.col('dataid') == params_list[0]['dataid'])['localminute'].max()}")
    pr(f"  solar dtype: {combined['solar'].dtype}")

    pr("\nDone.")


if __name__ == "__main__":
    main()
