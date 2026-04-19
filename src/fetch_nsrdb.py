"""
fetch_nsrdb.py
--------------
Pull hourly solar/weather data from the NREL NSRDB GOES CONUS PSM v4 API for
Austin, TX (lat 30.2672, lon -97.7431) for the year 2022 and save to data/raw/.

The old PSM3 endpoint was deprecated in 2024; this script uses the current
GOES-CONUS v4.0.0 endpoint which covers years 2018-present at up to 5-min
resolution.

Usage:
    python src/fetch_nsrdb.py

Requires:
    NREL_API_KEY set in .env (free key: https://developer.nrel.gov/signup/)
"""

import io
import os
import sys

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

LAT = 30.2672
LON = -97.7431
YEAR = "2022"
EMAIL = "cunningham.n@northeastern.edu"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(OUT_DIR, "nsrdb_austin_2022.csv")

# NSRDB GOES CONUS PSM v4 endpoint (replaced deprecated PSM3 in 2024)
# Covers continental US, years 2018–present, up to 5-min interval.
BASE_URL = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-conus-v4-0-0-download.csv"
)

ATTRIBUTES = (
    "ghi,dni,dhi,air_temperature,wind_speed,"
    "relative_humidity,surface_pressure,dew_point"
)

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_nsrdb() -> pd.DataFrame:
    params = {
        "api_key": API_KEY,
        "wkt": f"POINT({LON} {LAT})",
        "names": YEAR,
        "interval": "60",
        "attributes": ATTRIBUTES,
        "leap_day": "false",
        "utc": "false",
        "email": EMAIL,
        "full_name": "Nathan Cunningham",
        "affiliation": "Northeastern University",
        "reason": "research",
    }

    print(f"Requesting NSRDB GOES-CONUS v4 data for ({LAT}, {LON}), year {YEAR} ...")
    response = requests.get(BASE_URL, params=params, timeout=120)

    if response.status_code == 400:
        sys.exit(f"ERROR: Bad request (HTTP 400)\n{response.text[:500]}")
    if response.status_code == 403:
        sys.exit("ERROR: API key invalid or unauthorized (HTTP 403).")
    if response.status_code != 200:
        sys.exit(
            f"ERROR: NSRDB API returned HTTP {response.status_code}\n"
            f"{response.text[:500]}"
        )

    # Response CSV layout:
    #   Row 0 — metadata column names (Source, Location ID, City, ...)
    #   Row 1 — metadata values      (NSRDB, 1542084, ...)
    #   Row 2 — data column headers  (Year, Month, Day, Hour, Minute, GHI, ...)
    #   Row 3+ — hourly data records
    lines = response.text.splitlines()
    meta_keys = lines[0].split(",")
    meta_vals = lines[1].split(",")
    site_meta = dict(zip(meta_keys, meta_vals))
    print(f"Site: lat={site_meta.get('Latitude')}, "
          f"lon={site_meta.get('Longitude')}, "
          f"elevation={site_meta.get('Elevation')} m, "
          f"tz={site_meta.get('Time Zone')}")

    # Parse from row 2 onwards (column headers + data)
    data_csv = "\n".join(lines[2:])
    df = pd.read_csv(io.StringIO(data_csv))

    # Build a proper datetime index
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour"]].astype(str).agg("-".join, axis=1),
        format="%Y-%m-%d-%H",
    )
    df = df.set_index("datetime").drop(columns=["Year", "Month", "Day", "Hour", "Minute"])

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = fetch_nsrdb()

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
