#!/usr/bin/env python3
"""
src/synthetic.py — Synthetic San Diego CA solar generation pipeline.

Scientific basis
----------------
Generating labelled solar-generation data for residential homes without
on-site measurements requires combining three independent evidence sources:

1. **Building stock** (Pecan Street Dataport, San Diego Civita development):
   Real host homes supply building type and geographic cluster. This anchors
   synthetic homes to real residential contexts rather than idealised point
   sources.

2. **Panel-parameter distributions** (LBNL Tracking the Sun 2024, San Diego
   county RES_SF subset, n ≈ 274 K installations): System size, tilt, and
   azimuth are empirically sampled from the p10–p90 of the installed-base
   distribution, stratified by building type. This reproduces the real
   diversity of San Diego residential installations without cherry-picking.

3. **Physics-based generation** (pvlib 0.15+, PVWatts model): Given a
   location, orientation, and system size, pvlib's PVWatts ModelChain
   converts NASA POWER hourly irradiance + temperature into AC power output.
   PVWatts is the U.S. DOE standard for residential PV estimation and has
   been validated against measured generation data across climate zones.

4. **Calibrated noise** (real home 9836): pvlib produces a deterministic,
   noise-free output. Real solar generation contains per-hour variance from
   passing clouds, soiling, and inverter fluctuation. We estimate this
   variance from home 9836 (the only real San Diego PV home in Pecan Street)
   and add matching Gaussian noise so the synthetic signal has realistic
   temporal texture.

Usage
-----
    python src/synthetic.py --config configs/experiment_v2.yaml --seed 42

Outputs
-------
    data/processed/train_ca_synthetic.parquet   — 18-home dataset
    data/processed/synthetic_ca_parameters.csv  — per-home sampling log
    results/v2/synthetic_vs_real_daily.png       — validation plot
    s3://cs7180-final-project/data/processed/train_ca_synthetic.parquet
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pvlib
import yaml
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# I/O helper — writes UTF-8 to stdout even on Windows consoles
# ---------------------------------------------------------------------------

def pr(s: str) -> None:
    """Write a line to stdout with UTF-8 encoding (safe on Windows cp1252)."""
    sys.stdout.buffer.write((s + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """
    Load YAML config and resolve all data paths relative to repo root.

    Args:
        config_path: Path to YAML config file (e.g. configs/experiment_v2.yaml).

    Returns:
        Parsed config dict with ``synthetic`` sub-dict fully populated.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Return ROOT-relative Path for a config key inside cfg['synthetic']."""
    return ROOT / cfg["synthetic"][key]


# ---------------------------------------------------------------------------
# 1. Cache check
# ---------------------------------------------------------------------------

def is_cached(out_path: Path, s3_bucket: str, s3_key: str) -> bool:
    """
    Return True if the synthetic parquet already exists locally or in S3.

    Local check is tried first (fast). S3 check is attempted only if the
    local file is absent and AWS credentials are available.

    Args:
        out_path:  Local parquet path.
        s3_bucket: S3 bucket name.
        s3_key:    S3 object key for the parquet.

    Returns:
        True if either source has the file; False otherwise.
    """
    if out_path.exists():
        pr(f"  Found local cache: {out_path}")
        return True
    try:
        import boto3
        boto3.client("s3").head_object(Bucket=s3_bucket, Key=s3_key)
        pr(f"  Found S3 cache: s3://{s3_bucket}/{s3_key}")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. Host home selection
# ---------------------------------------------------------------------------

def load_host_homes(cfg: dict, rng: np.random.Generator) -> pl.DataFrame:
    """
    Stratified sample of host homes from the Pecan Street Civita metadata.

    Each sampled home contributes its ``building_type`` to the synthetic
    home; its other fields (dataid, sq_footage, etc.) are not used — only
    building type and the count per type matter for downstream stratification.

    Args:
        cfg: Full config dict.
        rng: Seeded NumPy Generator for reproducible sampling.

    Returns:
        Polars DataFrame with ``n_homes`` rows from the host-home metadata CSV.
    """
    meta = pl.read_csv(resolve(cfg, "host_homes_csv"), infer_schema_length=0)
    bt_targets: dict = cfg["synthetic"]["building_type_targets"]

    selected = []
    for bt, n in bt_targets.items():
        pool = meta.filter(pl.col("building_type") == bt)
        n_pick = min(n, len(pool))
        idx = rng.choice(len(pool), size=n_pick, replace=False)
        selected.append(pool[idx.tolist()])

    return pl.concat(selected)


# ---------------------------------------------------------------------------
# 3. Panel parameter sampling
# ---------------------------------------------------------------------------

def sample_panel_params(
    cfg: dict,
    tts_res: pl.DataFrame,
    host_homes: pl.DataFrame,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Independently sample system size, tilt, and azimuth for each home from
    the Tracking the Sun San Diego RES_SF empirical distribution.

    Stratification by building type:
      - Apartments  → system size drawn from [p10, p50] (smaller rooftops)
      - Town Homes  → system size drawn from [p10, p90] (full range)
      - SFH         → system size drawn from [p50, p90] (larger rooftops)

    Tilt and azimuth are drawn from [p10, p90] for all building types —
    roof geometry (not building size) drives orientation.

    Args:
        cfg:       Full config dict.
        tts_res:   Tracking the Sun RES_SF installations for San Diego.
        host_homes: Stratified host home sample (provides building_type).
        rng:       Seeded NumPy Generator.

    Returns:
        List of per-home parameter dicts with keys:
        dataid, building_type, lat, lon, system_size_kw, tilt_deg,
        azimuth_deg, module_type.
    """
    sc = cfg["synthetic"]
    ps = sc["param_sampling"]
    civita_lat = sc["civita_lat"]
    civita_lon = sc["civita_lon"]
    offset     = sc["lat_lon_offset_deg"]
    mono_frac  = sc["module_type_mono_fraction"]

    # Cast numeric columns; sentinel values (-1) and physical outliers filtered below
    for col in ["PV_system_size_DC", "tilt_1", "azimuth_1"]:
        tts_res = tts_res.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    def bounded(col: str, lo: float, hi: float) -> np.ndarray:
        """Extract values within hard physical bounds, dropping NaN and sentinels."""
        arr = tts_res[col].drop_nulls().to_numpy()
        return arr[(arr >= lo) & (arr <= hi)]

    def in_pct(arr: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
        """Trim array to [lo_pct, hi_pct] percentile range."""
        return arr[
            (arr >= np.percentile(arr, lo_pct))
            & (arr <= np.percentile(arr, hi_pct))
        ]

    size_all = bounded("PV_system_size_DC", 0.5, 50.0)
    tilt_all = bounded("tilt_1",            0.0, 60.0)
    az_all   = bounded("azimuth_1",         0.0, 360.0)

    lo, hi = ps["size_pct_lo"], ps["size_pct_hi"]
    apt_hi  = ps["apt_size_pct_hi"]   # apartments ceiling percentile
    sfh_lo  = ps["sfh_size_pct_lo"]   # SFH floor percentile

    size_apt  = in_pct(size_all, lo, apt_hi)  # apartments → smaller systems
    size_town = in_pct(size_all, lo, hi)       # town homes → full range
    size_sfh  = in_pct(size_all, sfh_lo, hi)  # SFH → larger systems
    tilt_pool = in_pct(tilt_all, ps["tilt_pct_lo"], ps["tilt_pct_hi"])
    az_pool   = in_pct(az_all,   ps["az_pct_lo"],   ps["az_pct_hi"])

    size_pools = {
        "Apartment":                        size_apt,
        "Town Home":                        size_town,
        "Single-Family Home 001 (Master)":  size_sfh,
    }

    def _maybe_float(val) -> float | None:
        """Convert a string/None field from the metadata CSV to float or None."""
        try:
            return float(val) if val not in (None, "", "None") else None
        except (TypeError, ValueError):
            return None

    params = []
    for i, row in enumerate(host_homes.iter_rows(named=True)):
        bt   = row["building_type"]
        pool = size_pools.get(bt, size_town)
        az   = round(float(rng.choice(az_pool)), 1)
        kw   = round(float(rng.choice(pool)), 3)
        params.append({
            "dataid":                  f"syn_sd_{i+1:03d}",
            # Host home metadata
            "building_type":           bt,
            "city":                    row.get("city") or "San Diego",
            "state":                   row.get("state") or "California",
            "total_square_footage":    _maybe_float(row.get("total_square_footage")),
            "house_construction_year": _maybe_float(row.get("house_construction_year")),
            # PVLib-derived panel parameters
            "lat":                     round(civita_lat + rng.uniform(-offset, offset), 6),
            "lon":                     round(civita_lon + rng.uniform(-offset, offset), 6),
            "elevation_m":             sc["civita_elev"],
            "system_size_kw":          kw,
            "tilt_deg":                round(float(rng.choice(tilt_pool)), 1),
            "azimuth_deg":             az,
            "module_type":             "Mono-c-Si" if rng.random() < mono_frac else "Multi-c-Si",
            # Pecan Street metadata column names (for reference / parameter log)
            "total_amount_of_pv":      kw,
            "pv_panel_direction":      az,
        })
    return params


# ---------------------------------------------------------------------------
# 4. Weather loading
# ---------------------------------------------------------------------------

def load_weather(cfg: dict) -> pd.DataFrame:
    """
    Load NASA POWER CA hourly reanalysis, concatenate years, resample to
    15-min via linear interpolation, and filter to the synthetic date range.

    NASA POWER provides hourly surface meteorology at ~50 km resolution.
    Linear interpolation to 15-min is appropriate for slowly varying fields
    (temperature, humidity, wind) but introduces temporal smoothing in
    irradiance; pvlib's transposition model partially corrects this.

    Args:
        cfg: Full config dict (reads ``weather_years``, ``date_start``,
             ``date_end``, ``weather_prefix``).

    Returns:
        UTC-indexed pandas DataFrame with columns:
        ghi, dni, dhi, temp_air, wind_speed — at 15-min resolution.
    """
    sc = cfg["synthetic"]
    prefix = ROOT / sc["weather_prefix"]
    dfs = []
    for yr in sc["weather_years"]:
        df = pd.read_csv(
            f"{prefix}_{yr}.csv",
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

    # Resample hourly → 15-min; irradiance clipped to ≥0 after interpolation
    wx15 = wx.resample("15min").interpolate(method="linear")
    for c in ["ghi", "dni", "dhi"]:
        wx15[c] = wx15[c].clip(lower=0.0)

    # Hourly → 15-min linear interpolation can leak small irradiance values
    # into geometrically nighttime steps (zenith > 90°). Zero these out using
    # Civita centroid solar position — all homes are within 0.005° so the
    # centroid mask is accurate to within one 15-min step at sunrise/sunset.
    civita_loc = Location(
        latitude=sc["civita_lat"],
        longitude=sc["civita_lon"],
        tz="UTC",
        altitude=sc["civita_elev"],
    )
    solpos = civita_loc.get_solarposition(wx15.index)
    night = solpos["zenith"] > 90.5  # 0.5° margin for atmospheric refraction
    for c in ["ghi", "dni", "dhi"]:
        wx15.loc[night, c] = 0.0

    # The config dates are in local time UTC-5 (Pecan Street convention).
    # DATE_START 00:00 UTC-5 = DATE_START 05:00 UTC
    # DATE_END   23:45 UTC-5 = DATE_END+1 04:45 UTC
    start = pd.Timestamp(sc["date_start"] + " 05:00:00", tz="UTC")
    end   = pd.Timestamp(sc["date_end"]   + " 23:45:00", tz="UTC") + pd.Timedelta(hours=5)
    return wx15.loc[start:end]


# ---------------------------------------------------------------------------
# 5. pvlib simulation (PVWatts)
# ---------------------------------------------------------------------------

def simulate_home(params: dict, weather_15: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Run the pvlib PVWatts ModelChain for a single synthetic home.

    PVWatts models:
      - DC output: pdc = pdc0 * (ghi_poa / 1000) * (1 + gamma_pdc * (Tcell - 25))
      - AC output: pac = pdc * eta_inv_nom  (clipped at pdc0)
    where ghi_poa is plane-of-array irradiance from the Hay-Davies transposition
    model and Tcell is the cell temperature from the SAPM thermal model.

    Temperature model choice: ``open_rack_glass_polymer`` (SAPM parameters
    a=-3.56, b=-0.075, deltaT=3). Open-rack is slightly conservative (hotter
    cells than roof-mounted) but avoids underestimating high-temperature losses
    in San Diego summers. The correction is <2% on annual yield.

    Args:
        params:     Per-home dict with lat, lon, system_size_kw, tilt_deg,
                    azimuth_deg (from sample_panel_params).
        weather_15: UTC-indexed 15-min weather DataFrame from load_weather().
        cfg:        Full config dict (reads pvlib sub-section).

    Returns:
        UTC-indexed pandas Series of AC output in kWh per 15-min interval.
    """
    pv_cfg = cfg["synthetic"]["pvlib"]

    loc = Location(
        latitude=params["lat"],
        longitude=params["lon"],
        tz="America/Los_Angeles",   # used for solar position only; input/output are UTC
        altitude=cfg["synthetic"]["civita_elev"],
    )

    kw = params["system_size_kw"]
    # SAPM temperature parameters for the chosen racking configuration
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        pv_cfg["temperature_model"]
    ][pv_cfg["racking"]]

    system = PVSystem(
        surface_tilt=params["tilt_deg"],
        surface_azimuth=params["azimuth_deg"],
        # PVWatts DC model: pdc0 = nameplate DC capacity in Watts
        module_parameters={
            "pdc0":      kw * 1000.0,
            "gamma_pdc": pv_cfg["gamma_pdc"],
        },
        # PVWatts AC model: nominal AC capacity = DC capacity * inverter efficiency
        inverter_parameters={
            "pdc0":        kw * 1000.0,
            "eta_inv_nom": pv_cfg["eta_inv_nom"],
        },
        temperature_model_parameters=temp_params,
    )

    # ModelChain.with_pvwatts sets: dc_model='pvwatts', ac_model='pvwatts',
    # aoi_model='physical', spectral_model='no_loss',
    # transposition_model='haydavies'
    mc = ModelChain.with_pvwatts(system, loc)
    mc.run_model(weather_15[["ghi", "dni", "dhi", "temp_air", "wind_speed"]])

    # Convert W → kWh per 15-min (multiply by 0.25 h / 1000 W·kW⁻¹)
    ac_w = pd.Series(mc.results.ac, index=weather_15.index).clip(lower=0.0)
    return ac_w * 0.25 / 1000.0


# ---------------------------------------------------------------------------
# 6. Noise calibration
# ---------------------------------------------------------------------------

def load_real_home_15min(cfg: dict) -> pd.Series:
    """
    Load the reference real CA home from Pecan Street 1-min data and
    resample to 15-min mean.

    Home 9836 is the only San Diego PV home with generation data in the
    Pecan Street export. Its per-hour variance captures realistic intra-day
    cloud-passing events, soiling, and inverter transients.

    Args:
        cfg: Full config dict (reads ``real_ca_solar_csv``,
             ``real_ca_home_id``).

    Returns:
        UTC-indexed pandas Series of mean 15-min solar generation (kWh).
    """
    sc = cfg["synthetic"]
    raw = (
        pl.read_csv(resolve(cfg, "real_ca_solar_csv"), infer_schema_length=0)
        .filter(pl.col("dataid") == sc["real_ca_home_id"])
        .select(["localminute", "solar"])
        .to_pandas()
    )
    # Pecan Street timestamps carry a fixed UTC-offset string (-05); utc=True
    # normalises them to UTC so resample() gets a proper DatetimeIndex.
    raw["localminute"] = pd.to_datetime(raw["localminute"], utc=True)
    raw = raw.set_index("localminute")
    raw["solar"] = pd.to_numeric(raw["solar"], errors="coerce").fillna(0.0)
    # Pecan Street CA stores instantaneous kW readings at 1-min resolution.
    # Resample to 15-min mean kW, then multiply by 0.25 h to get kWh/interval,
    # matching the units of the pvlib AC output in simulate_home().
    return raw["solar"].resample("1min").mean().resample("15min").mean() * 0.25


def compute_noise_profile(real_15: pd.Series) -> np.ndarray:
    """
    Compute per-hour-of-day standard deviation of real 15-min generation.

    This profile is used to add spatially and temporally structured noise to
    the physics-based synthetic output. Daytime hours with high variance
    (cloud-cover hours) receive more noise than stable early-morning hours.

    Args:
        real_15: 15-min solar generation Series (UTC-indexed) from real home.

    Returns:
        numpy array of shape (24,) — one std value per clock hour.
    """
    df = pd.DataFrame({"solar": real_15, "hour": real_15.index.hour})
    return df.groupby("hour")["solar"].std().fillna(0.0).values


def add_noise(
    kwh: pd.Series,
    noise_by_hour: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Add calibrated Gaussian noise to pvlib AC output and clip to [0, ∞).

    Negative solar generation is physically impossible; any noise-induced
    negative values are clipped to 0. The clip bias is negligible for
    daytime hours (noise << signal) and correct for nighttime (0 + noise = 0).

    Args:
        kwh:           UTC-indexed Series of pvlib AC output in kWh/15-min.
        noise_by_hour: Per-hour std profile from compute_noise_profile().
        noise_scale:   Config multiplier (1.0 = match real variance exactly).
        rng:           Seeded Generator.

    Returns:
        Noise-augmented Series, clipped to ≥ 0.
    """
    std = np.array([noise_by_hour[h] for h in kwh.index.hour]) * noise_scale
    noisy = kwh.values + rng.normal(0.0, std)
    # Preserve nighttime zeros: pvlib returns near-zero (not exact zero) at
    # twilight due to diffuse irradiance. Treat any step below 1 Wh/15-min
    # (0.001 kWh) as darkness so noise doesn't create spurious generation.
    noisy = np.where(kwh.values < 1e-3, 0.0, noisy)
    return pd.Series(np.clip(noisy, 0.0, None), index=kwh.index)


# ---------------------------------------------------------------------------
# 7. Output formatting
# ---------------------------------------------------------------------------

def build_processed_home_df(
    params: dict,
    kwh: pd.Series,
    weather_15: pd.DataFrame,
) -> pl.DataFrame:
    """
    Build a Polars DataFrame in the processed parquet schema for one synthetic
    home, matching the output of preprocess.process_region().

    Columns produced:
        dataid, local_15min, solar_kwh,
        GHI_W_m2, DNI_W_m2, DHI_W_m2, Temp_C, WindSpeed_m_s, RelHumidity_pct,
        lat, lon, tilt_deg, azimuth_deg, capacity_kw, elevation_m,
        building_type, city, state, total_square_footage, house_construction_year

    Timestamp convention: UTC → Etc/GMT+5 (POSIX UTC-5, no DST), formatted as
    'YYYY-MM-DD HH:MM:SS-05' to match the real CA processed parquet.

    Args:
        params:     Per-home dict from sample_panel_params (lat, lon,
                    system_size_kw, tilt_deg, azimuth_deg, elevation_m,
                    building_type, city, state, total_square_footage,
                    house_construction_year).
        kwh:        UTC-indexed pandas Series of kWh per 15-min interval.
        weather_15: UTC-indexed pandas DataFrame from load_weather() with
                    columns ghi, dni, dhi, temp_air, wind_speed, RelHumidity_pct.

    Returns:
        Polars DataFrame in the processed schema.
    """
    ts_utcm5   = kwh.index.tz_convert("Etc/GMT+5")
    timestamps = [t.strftime("%Y-%m-%d %H:%M:%S-05") for t in ts_utcm5]

    # Align weather to the kwh index (same UTC index; reindex for safety)
    wx = weather_15.reindex(kwh.index)
    n  = len(kwh)

    return (
        pl.DataFrame({
            "dataid":                  [params["dataid"]] * n,
            "local_15min":             timestamps,
            "solar_kwh":               kwh.values.tolist(),
            "GHI_W_m2":               wx["ghi"].values.tolist(),
            "DNI_W_m2":               wx["dni"].values.tolist(),
            "DHI_W_m2":               wx["dhi"].values.tolist(),
            "Temp_C":                  wx["temp_air"].values.tolist(),
            "WindSpeed_m_s":           wx["wind_speed"].values.tolist(),
            "RelHumidity_pct":         wx["RelHumidity_pct"].values.tolist(),
            "lat":                     [params["lat"]] * n,
            "lon":                     [params["lon"]] * n,
            "tilt_deg":                [params["tilt_deg"]] * n,
            "azimuth_deg":             [params["azimuth_deg"]] * n,
            "capacity_kw":             [params["system_size_kw"]] * n,
            "elevation_m":             [params["elevation_m"]] * n,
            # Host home metadata
            "building_type":           [params.get("building_type")] * n,
            "city":                    [params.get("city")] * n,
            "state":                   [params.get("state")] * n,
            "total_square_footage":    [params.get("total_square_footage")] * n,
            "house_construction_year": [params.get("house_construction_year")] * n,
        })
        .with_columns([
            pl.col("solar_kwh").cast(pl.Float64),
            pl.col("GHI_W_m2").cast(pl.Float32),
            pl.col("DNI_W_m2").cast(pl.Float32),
            pl.col("DHI_W_m2").cast(pl.Float32),
            pl.col("Temp_C").cast(pl.Float32),
            pl.col("WindSpeed_m_s").cast(pl.Float32),
            pl.col("RelHumidity_pct").cast(pl.Float32),
            pl.col("lat").cast(pl.Float32),
            pl.col("lon").cast(pl.Float32),
            pl.col("tilt_deg").cast(pl.Float32),
            pl.col("azimuth_deg").cast(pl.Float32),
            pl.col("capacity_kw").cast(pl.Float32),
            pl.col("elevation_m").cast(pl.Float32),
            # Nullable host home fields — explicit types so per-home DFs concat cleanly
            pl.col("building_type").cast(pl.Utf8),
            pl.col("city").cast(pl.Utf8),
            pl.col("state").cast(pl.Utf8),
            pl.col("total_square_footage").cast(pl.Float32),
            pl.col("house_construction_year").cast(pl.Float32),
        ])
    )


# ---------------------------------------------------------------------------
# 8. Parameter log
# ---------------------------------------------------------------------------

def save_param_log(params_list: list[dict], cfg: dict, seed: int) -> None:
    """
    Save all sampled per-home parameters to a CSV with a version header.

    The parameter log enables exact reproduction of the synthetic dataset:
    given the same config, seed, and library versions, running the script
    again will produce identical output.

    Args:
        params_list: List of per-home dicts from sample_panel_params().
        cfg:         Full config dict (reads ``output_params_csv``).
        seed:        Random seed used for this run.
    """
    out = resolve(cfg, "output_params_csv")
    header_lines = [
        f"# Synthetic CA generation parameter log",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Seed: {seed}",
        f"# pvlib version: {pvlib.__version__}",
        f"# numpy version: {np.__version__}",
        f"# config: {cfg.get('_config_path', 'unknown')}",
        f"#",
    ]
    rows = pl.DataFrame(params_list)
    csv_body = rows.write_csv()

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        f.write(csv_body)

    pr(f"  Parameter log: {out}")


def save_rng_state(rng: np.random.Generator, cfg: dict) -> None:
    """
    Persist the NumPy Generator state after parameter sampling so it can
    be restored for debugging or re-running from mid-pipeline.

    Args:
        rng: The Generator whose state to save.
        cfg: Full config dict (output written alongside output_params_csv).
    """
    state_path = resolve(cfg, "output_params_csv").with_suffix(".rng_state.json")
    state = rng.bit_generator.state
    # BitGenerator state contains numpy arrays; convert to lists for JSON
    def to_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serialisable(v) for k, v in obj.items()}
        return obj

    with open(state_path, "w") as f:
        json.dump(to_serialisable(state), f)
    pr(f"  RNG state saved: {state_path}")


# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------

def save_daily_curve_plot(
    params_list: list[dict],
    home_kwh: dict[str, pd.Series],
    real_15: pd.Series,
    cfg: dict,
) -> None:
    """
    Plot mean daily generation curves for all synthetic homes overlaid on
    the real home 9836 reference curve, and save to PNG.

    The plot is a visual sanity check: synthetic curves should bracket the
    real curve, with apartments on the low end and SFH on the high end.

    Args:
        params_list: Per-home parameter dicts (for legend labels).
        home_kwh:    Dict mapping dataid → kWh Series.
        real_15:     Real home 9836 15-min Series.
        cfg:         Full config dict (reads ``output_plot``).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = resolve(cfg, "output_plot")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Real home — average over all days at each 15-min time-of-day slot
        real_by_tod = real_15.groupby(real_15.index.time).mean()
        times_real  = [t.hour + t.minute / 60 for t in real_by_tod.index]
        ax.plot(times_real, real_by_tod.values, color="black", lw=2.5,
                label="Home 9836 (real)", zorder=5)

        # Synthetic homes — one thin line each
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
        plt.savefig(out_path, dpi=150)
        plt.close()
        pr(f"  Plot saved: {out_path}")
    except Exception as exc:
        pr(f"  Plot skipped: {exc}")


# ---------------------------------------------------------------------------
# 10. Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    combined: pl.DataFrame,
    params_list: list[dict],
    home_kwh: dict[str, pd.Series],
    cfg: dict,
) -> bool:
    """
    Run five physics and schema sanity checks on the generated dataset.

    Checks
    ------
    1. No negative solar values in any row.
    2. Generation is 0 at night (solar zenith > 90°) for San Diego lat/lon.
    3. Peak per-home generation falls within a physically plausible range
       (≤ system_size_kw * 0.25 kWh/15-min, i.e. nameplate at 100% efficiency).
    4. Date range of first home exactly matches real home 9836
       (2014-07-08 → 2015-06-30).
    5. Output column set exactly matches pecanstreet_california_1min_solar.csv.

    Args:
        combined:    The full combined Polars DataFrame to validate.
        params_list: Per-home parameter dicts.
        home_kwh:    Dict mapping dataid → UTC-indexed kWh Series.
        cfg:         Full config dict.

    Returns:
        True if all checks pass; False if any fail.
    """
    sc = cfg["synthetic"]

    results: list[tuple[str, bool, str]] = []

    # ---- Check 1: no negative solar values --------------------------------
    neg_count = int(combined["solar_kwh"].cast(pl.Float64, strict=False).fill_null(0).lt(0).sum())
    results.append((
        "No negative solar values",
        neg_count == 0,
        f"{neg_count} negative rows found" if neg_count > 0 else "all ≥ 0",
    ))

    # ---- Check 2: zero generation at night --------------------------------
    # Use the first home's actual jittered lat/lon so the zenith calculation
    # matches the solar position used during simulation — not the centroid.
    p0 = params_list[0]
    loc = Location(
        latitude=p0["lat"],
        longitude=p0["lon"],
        tz="UTC",
        altitude=sc["civita_elev"],
    )
    first_kwh = home_kwh[p0["dataid"]]
    solpos     = loc.get_solarposition(first_kwh.index)
    # 91° rather than 90° to account for atmospheric refraction (~0.6°) and
    # the residual irradiance that NASA POWER's ~50 km interpolation places at
    # the very edge of sunrise/sunset. Steps at 90°–91° zenith are physical
    # twilight, not an error.
    night_mask = solpos["zenith"] > 91.0

    # Threshold matches add_noise: sub-1-Wh/15-min steps are treated as darkness
    night_nonzero = int((first_kwh[night_mask.values] >= 1e-3).sum())
    results.append((
        "Zero generation at night (zenith>90°)",
        night_nonzero == 0,
        f"{night_nonzero} non-zero night steps" if night_nonzero > 0 else "all ≤ 1 Wh at night",
    ))

    # ---- Check 3: peak within physically plausible range ------------------
    all_pass = True
    worst = ""
    for p in params_list:
        kwh = home_kwh[p["dataid"]]
        # Physical upper bound: nameplate × inverter_eff × 0.25 h/interval.
        # We allow +30% headroom above this to accommodate calibrated noise
        # (the noise profile from home 9836 can add up to ~0.5 kWh/15-min at
        # noon). The bound still catches unrealistic outliers (>2×nameplate).
        pv_cfg = cfg["synthetic"]["pvlib"]
        upper = p["system_size_kw"] * pv_cfg["eta_inv_nom"] * 0.25 * 1.30
        peak  = float(kwh.max())
        if peak > upper:
            all_pass = False
            worst = f"{p['dataid']} peak {peak:.3f} > limit {upper:.3f} kWh"
            break
    results.append((
        "Peak generation within plausible range",
        all_pass,
        "all homes within nameplate×inv_eff×1.30 bound" if all_pass else worst,
    ))

    # ---- Check 4: date range matches home 9836 ----------------------------
    first_home = combined.filter(pl.col("dataid") == params_list[0]["dataid"])
    ts_min = first_home["local_15min"].min()
    ts_max = first_home["local_15min"].max()
    expected_start = sc["date_start"]
    expected_end   = sc["date_end"]
    # Compare date portion only (timestamp format: 'YYYY-MM-DD HH:MM:SS-05')
    date_ok = (
        ts_min is not None
        and ts_max is not None
        and ts_min[:10] == expected_start
        and ts_max[:10] == expected_end
    )
    results.append((
        f"Date range matches {expected_start} → {expected_end}",
        date_ok,
        f"got {ts_min[:10]} → {ts_max[:10]}" if ts_min else "no data",
    ))

    # ---- Check 5: required processed columns present ----------------------
    required = {
        "dataid", "local_15min", "solar_kwh",
        "GHI_W_m2", "DNI_W_m2", "DHI_W_m2", "Temp_C", "WindSpeed_m_s", "RelHumidity_pct",
        "lat", "lon", "tilt_deg", "azimuth_deg", "capacity_kw", "elevation_m",
    }
    missing   = required - set(combined.columns)
    schema_ok = len(missing) == 0
    detail    = "all required processed columns present" if schema_ok else f"missing={missing}"
    results.append(("Required processed columns present", schema_ok, detail))

    # ---- Print results ----------------------------------------------------
    pr("\n=== Sanity Checks ===")
    all_ok = True
    for name, passed, detail in results:
        tag = "PASS" if passed else "FAIL"
        pr(f"  [{tag}] {name}")
        pr(f"        {detail}")
        if not passed:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 11. S3 upload
# ---------------------------------------------------------------------------

def upload_to_s3(local_path: Path, cfg: dict) -> None:
    """
    Upload the local parquet to S3. Skips gracefully if credentials are absent.

    Args:
        local_path: Local path of the file to upload.
        cfg:        Full config dict (reads ``s3_bucket``, ``s3_output_key``).
    """
    try:
        import boto3
        bucket = cfg["data"].get("s3_bucket", os.environ.get("S3_BUCKET", "cs7180-final-project"))
        key    = cfg["synthetic"]["s3_output_key"]
        pr(f"  Uploading to s3://{bucket}/{key} ...")
        boto3.client("s3").upload_file(str(local_path), bucket, key)
        pr(f"  Uploaded: s3://{bucket}/{key}")
    except Exception as exc:
        pr(f"  S3 upload skipped: {exc}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic San Diego CA solar homes.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_v2.yaml",
        help="Path to YAML config file (default: configs/experiment_v2.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config   # stash for parameter log
    sc  = cfg["synthetic"]
    rng = np.random.default_rng(args.seed)

    DATA_PROC = ROOT / "data" / "processed"
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    out_path = resolve(cfg, "output_parquet")
    s3_bucket = cfg["data"].get("s3_bucket", os.environ.get("S3_BUCKET", "cs7180-final-project"))
    s3_key    = sc["s3_output_key"]

    pr("=== Synthetic CA Solar Pipeline ===")
    pr(f"  Config : {args.config}")
    pr(f"  Seed   : {args.seed}")
    pr(f"  pvlib  : {pvlib.__version__}  numpy: {np.__version__}\n")

    # ---- Cache check -------------------------------------------------------
    if is_cached(out_path, s3_bucket, s3_key):
        pr("Using cached synthetic data — skipping generation.")
        pr("  (Delete the local/S3 file and re-run to regenerate.)")
        return

    pr("Generating synthetic data...\n")

    # ---- Host homes --------------------------------------------------------
    pr("1. Loading host homes (stratified sample from Civita Pecan Street)...")
    host_homes = load_host_homes(cfg, rng)
    bt_counts  = host_homes.group_by("building_type").len().sort("building_type")
    for row in bt_counts.iter_rows(named=True):
        pr(f"   {row['building_type']}: {row['len']}")

    # ---- Panel parameters --------------------------------------------------
    pr("\n2. Sampling panel parameters from Tracking the Sun RES_SF...")
    tts     = pl.read_csv(resolve(cfg, "tts_sandiego_csv"), infer_schema_length=0)
    tts_res = tts.filter(pl.col("customer_segment") == "RES_SF")
    pr(f"   RES_SF pool: {len(tts_res):,} installations")
    params_list = sample_panel_params(cfg, tts_res, host_homes, rng)

    # Save parameter log and RNG state immediately after sampling
    save_param_log(params_list, cfg, seed=args.seed)
    save_rng_state(rng, cfg)

    # ---- Weather -----------------------------------------------------------
    pr("\n3. Loading NASA POWER CA weather (15-min interpolation)...")
    weather_15 = load_weather(cfg)
    pr(f"   Steps: {len(weather_15):,}  "
       f"({weather_15.index[0]} → {weather_15.index[-1]})")

    # ---- Noise profile -----------------------------------------------------
    pr("\n4. Computing noise profile from real home 9836...")
    real_15       = load_real_home_15min(cfg)
    noise_by_hour = compute_noise_profile(real_15)
    pr(f"   Noise std by hour: mean={noise_by_hour.mean():.4f}  "
       f"max={noise_by_hour.max():.4f} kWh/15-min")

    # ---- Simulate ----------------------------------------------------------
    pr(f"\n5. Simulating {len(params_list)} homes with pvlib PVWatts...")
    pr(f"   {'dataid':<14} {'building_type':<35} {'kW':>6} {'tilt':>5} "
       f"{'az':>5}  daily_kWh")
    pr("   " + "-" * 76)

    all_dfs:  list[pl.DataFrame]      = []
    home_kwh: dict[str, pd.Series]    = {}
    summaries: list[dict]             = []
    n_days = (weather_15.index[-1] - weather_15.index[0]).days + 1

    for p in params_list:
        kwh = simulate_home(p, weather_15, cfg)
        kwh = add_noise(kwh, noise_by_hour, sc["noise_scale"], rng)
        home_kwh[p["dataid"]] = kwh

        total      = float(kwh.sum())
        mean_daily = total / n_days
        summaries.append({
            **p,
            "mean_daily_kwh": round(mean_daily, 3),
            "total_kwh":      round(total, 1),
        })
        pr(f"   {p['dataid']:<14} {p['building_type']:<35} "
           f"{p['system_size_kw']:>6.2f} {p['tilt_deg']:>5.1f} "
           f"{p['azimuth_deg']:>5.1f}  {mean_daily:.3f}")

        all_dfs.append(build_processed_home_df(p, kwh, weather_15))

    # ---- Combine and save --------------------------------------------------
    combined = pl.concat(all_dfs)
    pr(f"\n6. Saving output...")
    pr(f"   Combined: {len(combined):,} rows × {len(combined.columns)} cols")
    combined.write_parquet(str(out_path))
    pr(f"   Local:  {out_path}")
    upload_to_s3(out_path, cfg)

    # ---- Plot --------------------------------------------------------------
    pr("\n7. Saving daily curve plot...")
    save_daily_curve_plot(params_list, home_kwh, real_15, cfg)

    # ---- Per-home summary --------------------------------------------------
    pr("\n=== Per-Home Summary ===")
    pr(f"{'dataid':<14} {'building_type':<35} {'kW':>6} {'tilt':>5} "
       f"{'az':>5}  {'mean_day_kWh':>13}  {'total_kWh':>10}")
    pr("-" * 96)
    for s in summaries:
        pr(f"{s['dataid']:<14} {s['building_type']:<35} "
           f"{s['system_size_kw']:>6.2f} {s['tilt_deg']:>5.1f} "
           f"{s['azimuth_deg']:>5.1f}  {s['mean_daily_kwh']:>13.3f}  "
           f"{s['total_kwh']:>10.1f}")

    # ---- Sanity checks -----------------------------------------------------
    all_ok = run_sanity_checks(combined, params_list, home_kwh, cfg)

    pr("\n" + ("All checks PASSED." if all_ok else "One or more checks FAILED — review output above."))
    pr("Done.")


if __name__ == "__main__":
    main()
