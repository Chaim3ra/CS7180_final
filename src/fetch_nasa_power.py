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

import pandas as pd
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
    # Austin, TX
    {"name": "austin",     "lat": 30.2672,  "lon": -97.7431,  "year": 2018},
    # New York, NY
    {"name": "newyork",    "lat": 40.7128,  "lon": -74.0060,  "year": 2019},
    # San Jose, CA — NASA POWER covers all years (unlike NSRDB GOES which starts 2018)
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2014},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2015},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2016},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2017},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2018},
]


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_one(lat: float, lon: float, year: int) -> pd.DataFrame:
    """
    Fetch a full calendar year of hourly data from NASA POWER for one location.

    Returns a DataFrame with a DatetimeIndex (LST) and one column per parameter.
    Missing-data fill values (-999.0) are replaced with NaN.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "start":         f"{year}0101",
        "end":           f"{year}1231",
        "community":     "RE",          # Renewable Energy community
        "parameters":    PARAMETERS,
        "format":        "JSON",
        "header":        "true",
        "time-standard": "LST",         # local standard time
    }

    response = requests.get(BASE_URL, params=params, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} — {response.text[:300]}"
        )

    payload = response.json()

    # Check for API-level error messages
    messages = payload.get("messages", [])
    if messages:
        raise RuntimeError(f"API error: {messages}")

    # Build DataFrame: each key in parameter dict becomes a column;
    # index is the YYYYMMDDHH timestamp string.
    param_data = payload["properties"]["parameter"]
    df = pd.DataFrame(param_data)           # index = YYYYMMDDHH strings, cols = params

    # Parse YYYYMMDDHH → datetime
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "datetime"

    # Replace fill values with NaN
    df.replace(FILL_VALUE, float("nan"), inplace=True)

    # Rename columns to friendlier names
    df.rename(columns={
        "ALLSKY_SFC_SW_DWN":  "GHI_W_m2",
        "ALLSKY_SFC_SW_DNI":  "DNI_W_m2",
        "ALLSKY_SFC_SW_DIFF": "DHI_W_m2",
        "T2M":                "Temp_C",
        "WS2M":               "WindSpeed_m_s",
        "RH2M":               "RelHumidity_pct",
    }, inplace=True)

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

        # --- Cache check ---
        if os.path.exists(fpath):
            print(f"  SKIP — already cached ({os.path.getsize(fpath):,} bytes)")
            results["cached"].append(fname)
            continue

        # --- API fetch ---
        print(f"  Fetching ({lat}, {lon}), year {year} ...")
        try:
            df = fetch_one(lat, lon, year)
            df.to_csv(fpath)
            print(f"  OK — {len(df):,} rows, {df.shape[1]} cols -> {fname}")
            results["fetched"].append(fname)
        except Exception as exc:
            print(f"  ERROR — {exc}")
            results["failed"].append(fname)

        # Polite delay between requests
        if i < len(TARGETS) - 1:
            time.sleep(REQUEST_DELAY_S)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if results["fetched"]:
        print(f"  Fetched ({len(results['fetched'])}): {', '.join(results['fetched'])}")
    if results["cached"]:
        print(f"  Cached  ({len(results['cached'])}): {', '.join(results['cached'])}")
    if results["failed"]:
        print(f"  Failed  ({len(results['failed'])}): {', '.join(results['failed'])}")

    # Preview the first fetched file
    fetched_paths = [
        os.path.join(OUT_DIR, f)
        for f in results["fetched"]
        if os.path.exists(os.path.join(OUT_DIR, f))
    ]
    if fetched_paths:
        sample = pd.read_csv(fetched_paths[0], index_col=0, parse_dates=True)
        print(f"\nPreview — {os.path.basename(fetched_paths[0])}:")
        print(sample.head(5).to_string())


if __name__ == "__main__":
    main()
