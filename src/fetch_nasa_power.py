"""
fetch_nasa_power.py
-------------------
Fetch hourly solar/weather data from the NASA POWER API for all project
location/year combinations and cache each result as a separate CSV in data/raw/.
Skips the API call if the output file already exists.

No API key required.

API docs: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/

Usage:
    python src/fetch_nasa_power.py

Parameters fetched:
    ALLSKY_SFC_SW_DWN  — All-sky surface shortwave downwelling irradiance (GHI), W/m²
    ALLSKY_SFC_SW_DNI  — All-sky surface shortwave direct normal irradiance (DNI), W/m²
    ALLSKY_SFC_SW_DIFF — All-sky surface shortwave diffuse irradiance (DHI), W/m²
    T2M                — Temperature at 2 m, °C
    WS2M               — Wind speed at 2 m, m/s
    RH2M               — Relative humidity at 2 m, %
"""

import os
import time

import polars as pl
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

PARAMETERS = "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS2M,RH2M"
FILL_VALUE = -999.0       # NASA POWER sentinel for missing data
REQUEST_DELAY_S = 1.0     # polite gap between API calls

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ---------------------------------------------------------------------------
# Fetch targets  (name, lat, lon, year)
# Output filename: nasa_power_{name}_{year}.csv
# ---------------------------------------------------------------------------
TARGETS = [
    {"name": "austin",     "lat": 30.2672,  "lon": -97.7431,  "year": 2018},
    {"name": "newyork",    "lat": 40.7128,  "lon": -74.0060,  "year": 2019},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2014},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2015},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2016},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2017},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2018},
]


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_one(lat: float, lon: float, year: int) -> pl.DataFrame:
    """Fetch a full calendar year of hourly data from NASA POWER.

    Args:
        lat: Latitude of the target location.
        lon: Longitude of the target location.
        year: Calendar year to fetch.

    Returns:
        Polars DataFrame with a ``datetime`` column and one column per
        parameter.  Missing-data fill values (``-999.0``) are replaced
        with ``null``.

    Raises:
        RuntimeError: On non-200 HTTP status or API-level error messages.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "start":         f"{year}0101",
        "end":           f"{year}1231",
        "community":     "RE",
        "parameters":    PARAMETERS,
        "format":        "JSON",
        "header":        "true",
        "time-standard": "LST",
    }

    response = requests.get(BASE_URL, params=params, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} — {response.text[:300]}"
        )

    payload = response.json()
    messages = payload.get("messages", [])
    if messages:
        raise RuntimeError(f"API error: {messages}")

    # param_data: {"PARAM_NAME": {"YYYYMMDDHH": value, ...}, ...}
    param_data = payload["properties"]["parameter"]
    timestamps = list(next(iter(param_data.values())).keys())

    data = {param: list(vals.values()) for param, vals in param_data.items()}
    df = pl.DataFrame(data).with_columns(
        pl.Series("datetime", timestamps)
        .str.to_datetime(format="%Y%m%d%H")
    )

    # Replace fill values with null
    data_cols = [c for c in df.columns if c != "datetime"]
    df = df.with_columns([
        pl.when(pl.col(c) == FILL_VALUE).then(None).otherwise(pl.col(c)).alias(c)
        for c in data_cols
    ])

    df = df.rename({
        "ALLSKY_SFC_SW_DWN":  "GHI_W_m2",
        "ALLSKY_SFC_SW_DNI":  "DNI_W_m2",
        "ALLSKY_SFC_SW_DIFF": "DHI_W_m2",
        "T2M":                "Temp_C",
        "WS2M":               "WindSpeed_m_s",
        "RH2M":               "RelHumidity_pct",
    })

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {"fetched": [], "cached": [], "failed": []}

    for i, target in enumerate(TARGETS):
        name  = target["name"]
        lat   = target["lat"]
        lon   = target["lon"]
        year  = target["year"]
        fname = f"nasa_power_{name}_{year}.csv"
        fpath = os.path.join(OUT_DIR, fname)

        print(f"\n[{i+1}/{len(TARGETS)}] {fname}")

        if os.path.exists(fpath):
            print(f"  SKIP — already cached ({os.path.getsize(fpath):,} bytes)")
            results["cached"].append(fname)
            continue

        print(f"  Fetching ({lat}, {lon}), year {year} ...")
        try:
            df = fetch_one(lat, lon, year)
            df.write_csv(fpath)
            print(f"  OK — {len(df):,} rows, {df.width} cols -> {fname}")
            results["fetched"].append(fname)
        except Exception as exc:
            print(f"  ERROR — {exc}")
            results["failed"].append(fname)

        if i < len(TARGETS) - 1:
            time.sleep(REQUEST_DELAY_S)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if results["fetched"]:
        print(f"  Fetched ({len(results['fetched'])}): {', '.join(results['fetched'])}")
    if results["cached"]:
        print(f"  Cached  ({len(results['cached'])}): {', '.join(results['cached'])}")
    if results["failed"]:
        print(f"  Failed  ({len(results['failed'])}): {', '.join(results['failed'])}")

    fetched_paths = [
        os.path.join(OUT_DIR, f)
        for f in results["fetched"]
        if os.path.exists(os.path.join(OUT_DIR, f))
    ]
    if fetched_paths:
        sample = pl.read_csv(fetched_paths[0])
        print(f"\nPreview — {os.path.basename(fetched_paths[0])}:")
        print(sample.head(5))


if __name__ == "__main__":
    main()
