"""
fetch_nsrdb.py
--------------
Pull hourly solar/weather data from the NREL NSRDB PSM3 API for Austin, TX
(lat 30.2672, lon -97.7431) for the year 2022 and save to data/raw/.

Usage:
    python src/fetch_nsrdb.py

Requires:
    NREL_API_KEY set in .env (get a free key at https://developer.nrel.gov/signup/)
"""

import os
import sys
import io

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("NREL_API_KEY")
if not API_KEY or API_KEY == "your_key_here":
    sys.exit(
        "ERROR: NREL_API_KEY is not set. "
        "Add your key to .env (get one free at https://developer.nrel.gov/signup/)"
    )

LAT = 30.2672
LON = -97.7431
YEAR = 2022
EMAIL = "user@example.com"  # NSRDB requires a contact email in the request

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(OUT_DIR, "nsrdb_austin_2022.csv")

# NSRDB PSM3 CSV endpoint
BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_nsrdb() -> pd.DataFrame:
    params = {
        "api_key": API_KEY,
        "lat": LAT,
        "lon": LON,
        "year": YEAR,
        "interval": 60,          # hourly
        "attributes": (
            "ghi,dni,dhi,air_temperature,wind_speed,"
            "relative_humidity,surface_pressure,dew_point"
        ),
        "leap_day": "false",
        "utc": "false",
        "email": EMAIL,
        "names": YEAR,
        "full_name": "CS7180+Project",
        "affiliation": "Northeastern+University",
        "reason": "research",
    }

    print(f"Requesting NSRDB data for ({LAT}, {LON}), year {YEAR} ...")
    response = requests.get(BASE_URL, params=params, timeout=120)

    if response.status_code != 200:
        sys.exit(
            f"ERROR: NSRDB API returned HTTP {response.status_code}\n{response.text[:500]}"
        )

    # The first two rows are metadata; row 3 onward is the actual data with a
    # two-row header (variable name + unit). We parse metadata separately.
    raw_text = response.text
    lines = raw_text.splitlines()

    # Row 0: site metadata (key-value pairs)
    meta = dict(zip(lines[0].split(","), lines[1].split(",")))
    print("Site metadata:", meta)

    # Rows 2+: data (first row = column names, second row = units, rest = data)
    data_text = "\n".join(lines[2:])
    df = pd.read_csv(io.StringIO(data_text), header=0, skiprows=[1])

    return df


def main():
    df = fetch_nsrdb()

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
