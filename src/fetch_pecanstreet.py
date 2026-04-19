"""
fetch_pecanstreet.py
--------------------
Authenticate with the Pecan Street Dataport API, pull a small sample of
residential solar generation data (a few homes, a few days), and save to
data/raw/pecanstreet_sample.csv.

Usage:
    python src/fetch_pecanstreet.py

Requires:
    PECAN_STREET_USERNAME and PECAN_STREET_PASSWORD set in .env
    (request access at https://dataport.pecanstreet.org)

Pecan Street Dataport uses a PostgreSQL-over-HTTP interface (PostgREST).
Authentication is done via a POST to /api/token which returns a JWT that must
be included as a Bearer token in subsequent requests.
"""

import os
import sys

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

USERNAME = os.getenv("PECAN_STREET_USERNAME")
PASSWORD = os.getenv("PECAN_STREET_PASSWORD")

if not USERNAME or USERNAME == "your_username_here":
    sys.exit(
        "ERROR: PECAN_STREET_USERNAME is not set. Add your credentials to .env\n"
        "       Request access at https://dataport.pecanstreet.org"
    )
if not PASSWORD or PASSWORD == "your_password_here":
    sys.exit(
        "ERROR: PECAN_STREET_PASSWORD is not set. Add your credentials to .env\n"
        "       Request access at https://dataport.pecanstreet.org"
    )

BASE_URL = "https://dataport.pecanstreet.org"
AUTH_ENDPOINT = f"{BASE_URL}/api/token"
DATA_ENDPOINT = f"{BASE_URL}/api/pge_egauge_minutes"   # 1-min interval table

# Sample window: a few homes, a few days
SAMPLE_HOMES = 5      # number of distinct dataid values to pull
SAMPLE_DAYS = 3       # number of days
START_DATE = "2022-06-01"
END_DATE = "2022-06-03"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(OUT_DIR, "pecanstreet_sample.csv")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_token() -> str:
    """POST credentials and return the JWT access token."""
    print("Authenticating with Pecan Street Dataport ...")
    try:
        resp = requests.post(
            AUTH_ENDPOINT,
            json={"username": USERNAME, "password": PASSWORD},
            timeout=30,
        )
    except requests.exceptions.ConnectionError as exc:
        sys.exit(f"ERROR: Could not reach {AUTH_ENDPOINT}\n  {exc}")

    if resp.status_code == 401:
        sys.exit(
            "ERROR: Authentication failed (HTTP 401). "
            "Check PECAN_STREET_USERNAME and PECAN_STREET_PASSWORD in .env."
        )
    if resp.status_code == 403:
        sys.exit(
            "ERROR: Access denied (HTTP 403). "
            "Your account may not have Dataport access yet — "
            "visit https://dataport.pecanstreet.org to request it."
        )
    if resp.status_code != 200:
        sys.exit(
            f"ERROR: Unexpected response from auth endpoint "
            f"(HTTP {resp.status_code})\n{resp.text[:500]}"
        )

    token = resp.json().get("token") or resp.json().get("access_token")
    if not token:
        sys.exit(f"ERROR: No token found in auth response: {resp.json()}")

    print("Authentication successful.")
    return token


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_sample(token: str) -> pd.DataFrame:
    """Pull a small slice of solar generation data using the bearer token."""
    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: discover a few valid dataids that have solar (solar column > 0)
    print(f"Fetching up to {SAMPLE_HOMES} home IDs with solar data ...")
    id_resp = requests.get(
        DATA_ENDPOINT,
        headers=headers,
        params={
            "select": "dataid",
            "solar": "gt.0",
            "localminute": f"gte.{START_DATE}T00:00:00",
            "limit": 500,
        },
        timeout=60,
    )
    if id_resp.status_code != 200:
        sys.exit(
            f"ERROR: Data query failed (HTTP {id_resp.status_code})\n{id_resp.text[:500]}"
        )

    ids = list({row["dataid"] for row in id_resp.json()})[:SAMPLE_HOMES]
    if not ids:
        sys.exit(
            "ERROR: No homes with solar data found in the queried range. "
            "Try adjusting START_DATE / END_DATE."
        )
    print(f"  Found homes: {ids}")

    # Step 2: pull minute-level solar generation for those homes
    print(f"Fetching minute-level solar data for {START_DATE} to {END_DATE} ...")
    data_resp = requests.get(
        DATA_ENDPOINT,
        headers=headers,
        params={
            "select": "dataid,localminute,solar",
            "dataid": f"in.({','.join(str(i) for i in ids)})",
            "localminute": f"gte.{START_DATE}T00:00:00",
            "localminute": f"lte.{END_DATE}T23:59:59",
            "order": "dataid,localminute",
            "limit": 50000,
        },
        timeout=120,
    )
    if data_resp.status_code != 200:
        sys.exit(
            f"ERROR: Data query failed (HTTP {data_resp.status_code})\n{data_resp.text[:500]}"
        )

    records = data_resp.json()
    if not records:
        sys.exit("ERROR: API returned 0 rows. Check the date range and column names.")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    token = get_token()
    df = fetch_sample(token)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
