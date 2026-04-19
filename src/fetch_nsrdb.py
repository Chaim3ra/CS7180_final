"""
fetch_nsrdb.py
--------------
Pull hourly solar/weather data from the NREL NSRDB GOES CONUS PSM v4 API for
multiple location/year combinations and cache each result as a separate CSV in
data/raw/.  A file is skipped (not re-fetched) if it already exists on disk.

The old PSM3 endpoint was deprecated in 2024; this script uses the current
GOES-CONUS v4.0.0 endpoint which covers continental US for years 2018-present.
Requests for years outside that range are skipped with a warning.

Usage:
    python src/fetch_nsrdb.py

Requires:
    NREL_API_KEY set in .env (free key: https://developer.nrel.gov/signup/)
"""

import io
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("NREL_API_KEY")
if not API_KEY or API_KEY == "your_key_here":
    sys.exit(
        "ERROR: NREL_API_KEY is not set. "
        "Add your key to .env (free at https://developer.nrel.gov/signup/)"
    )

EMAIL = "cunningham.n@northeastern.edu"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# NSRDB GOES CONUS PSM v4 endpoint (replaced deprecated PSM3 in 2024).
# Valid years: 2018–present (GOES-East satellite coverage).
BASE_URL = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-conus-v4-0-0-download.csv"
)
GOES_CONUS_MIN_YEAR = 2018

ATTRIBUTES = (
    "ghi,dni,dhi,air_temperature,wind_speed,"
    "relative_humidity,surface_pressure,dew_point"
)

# API rate limit: 1 req/sec.  Add a small buffer between calls.
REQUEST_DELAY_S = 1.1

# ---------------------------------------------------------------------------
# Fetch targets
# Each dict: name (used in filename), lat, lon, year
# Output filename pattern: nsrdb_{name}_{year}.csv
# ---------------------------------------------------------------------------
TARGETS = [
    # Austin, TX
    {"name": "austin",     "lat": 30.2672,  "lon": -97.7431,  "year": 2018},
    # New York, NY
    {"name": "newyork",    "lat": 40.7128,  "lon": -74.0060,  "year": 2019},
    # California (San Jose) — 2014-2017 pre-date GOES CONUS coverage and will be skipped
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2014},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2015},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2016},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2017},
    {"name": "california", "lat": 37.3382,  "lon": -121.8863, "year": 2018},
]


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_one(lat: float, lon: float, year: int) -> pd.DataFrame:
    """Fetch one location/year from the NSRDB API and return a DataFrame."""
    params = {
        "api_key": API_KEY,
        "wkt": f"POINT({lon} {lat})",
        "names": str(year),
        "interval": "60",
        "attributes": ATTRIBUTES,
        "leap_day": "false",
        "utc": "false",
        "email": EMAIL,
        "full_name": "Nathan Cunningham",
        "affiliation": "Northeastern University",
        "reason": "research",
    }

    response = requests.get(BASE_URL, params=params, timeout=120)

    if response.status_code == 400:
        raise ValueError(f"HTTP 400 — {response.text[:300]}")
    if response.status_code == 403:
        raise PermissionError("HTTP 403 — API key invalid or unauthorized.")
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} — {response.text[:300]}"
        )

    # Response CSV layout:
    #   Row 0 — metadata column names  (Source, Location ID, City, ...)
    #   Row 1 — metadata values        (NSRDB, 1542084, ...)
    #   Row 2 — data column headers    (Year, Month, Day, Hour, Minute, GHI, ...)
    #   Row 3+ — hourly data records
    lines = response.text.splitlines()
    meta = dict(zip(lines[0].split(","), lines[1].split(",")))
    print(f"       site: lat={meta.get('Latitude')}, lon={meta.get('Longitude')}, "
          f"elev={meta.get('Elevation')} m, tz={meta.get('Time Zone')}")

    df = pd.read_csv(io.StringIO("\n".join(lines[2:])))

    # Build a proper datetime index
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour"]].astype(str).agg("-".join, axis=1),
        format="%Y-%m-%d-%H",
    )
    df = df.set_index("datetime").drop(
        columns=["Year", "Month", "Day", "Hour", "Minute"]
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {"fetched": [], "cached": [], "skipped": [], "failed": []}

    for i, target in enumerate(TARGETS):
        name  = target["name"]
        lat   = target["lat"]
        lon   = target["lon"]
        year  = target["year"]
        fname = f"nsrdb_{name}_{year}.csv"
        fpath = os.path.join(OUT_DIR, fname)

        print(f"\n[{i+1}/{len(TARGETS)}] {fname}")

        # --- Cache check ---
        if os.path.exists(fpath):
            print(f"       SKIP — already cached ({os.path.getsize(fpath):,} bytes)")
            results["cached"].append(fname)
            continue

        # --- Year range guard ---
        if year < GOES_CONUS_MIN_YEAR:
            print(
                f"       SKIP — year {year} is before GOES CONUS v4 coverage "
                f"(2018-present). Pre-2018 CONUS data is not available via "
                f"this endpoint since PSM3 was deprecated."
            )
            results["skipped"].append(fname)
            continue

        # --- API fetch ---
        print(f"       Fetching ({lat}, {lon}), year {year} ...")
        try:
            df = fetch_one(lat, lon, year)
            df.to_csv(fpath)
            print(f"       OK — {len(df):,} rows saved to {fname}")
            results["fetched"].append(fname)
        except (ValueError, PermissionError, RuntimeError) as exc:
            print(f"       ERROR — {exc}")
            results["failed"].append(fname)

        # Respect API rate limit between requests
        if i < len(TARGETS) - 1:
            time.sleep(REQUEST_DELAY_S)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if results["fetched"]:
        print(f"  Fetched  ({len(results['fetched'])}): {', '.join(results['fetched'])}")
    if results["cached"]:
        print(f"  Cached   ({len(results['cached'])}): {', '.join(results['cached'])}")
    if results["skipped"]:
        print(f"  Skipped  ({len(results['skipped'])}): {', '.join(results['skipped'])}")
    if results["failed"]:
        print(f"  Failed   ({len(results['failed'])}): {', '.join(results['failed'])}")

    # Preview one of the fetched files
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
