"""
fetch_pvdaq.py
--------------
Download the PVDAQ systems metadata CSV from the DOE Open Energy Data Initiative
(OEDI) and print a summary: total systems, breakdown by state, and data
coverage stats.

No API key required — the file is publicly hosted on S3.

Source: https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20241231.csv
PVDAQ:  https://data.openei.org/submissions/4

Usage:
    python src/fetch_pvdaq.py
"""

import os

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
URL = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20241231.csv"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(OUT_DIR, "pvdaq_systems_metadata.csv")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download() -> pd.DataFrame:
    """Stream the PVDAQ systems CSV to disk and return it as a DataFrame."""
    print(f"Downloading PVDAQ systems metadata ...")
    response = requests.get(URL, stream=True, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} — {response.text[:300]}"
        )

    total = int(response.headers.get("content-length", 0))
    received = 0
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(OUT_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            received += len(chunk)

    print(f"  Saved {received:,} bytes to {OUT_PATH}")
    return pd.read_csv(OUT_PATH)


def load_or_download() -> pd.DataFrame:
    if os.path.exists(OUT_PATH):
        print(f"Cache hit — loading {OUT_PATH}")
        return pd.read_csv(OUT_PATH)
    return download()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print("PVDAQ Systems Metadata — Summary")
    print(f"{'=' * 60}")
    print(f"Total systems:  {len(df):,}")
    print(f"Columns:        {list(df.columns)}")

    # --- State breakdown ---
    # 'location' column is "City, ST" format — extract the 2-letter state code.
    if "location" in df.columns:
        df = df.copy()
        df["state"] = df["location"].str.extract(r",\s*([A-Z]{2})\s*$")
        state_counts = (
            df["state"]
            .value_counts()
            .rename_axis("state")
            .reset_index(name="systems")
        )
        print(f"\nSystems by state ({state_counts['state'].nunique()} states):")
        print(state_counts.to_string(index=False))
    else:
        print("\nWARN: 'location' column not found — skipping state breakdown.")

    # --- Data coverage ---
    if "years" in df.columns:
        yr = df["years"].dropna()
        print(f"\nData coverage (years per system):")
        print(f"  min:    {yr.min():.2f}")
        print(f"  median: {yr.median():.2f}")
        print(f"  max:    {yr.max():.2f}")

    if "data_points_count" in df.columns:
        dp = df["data_points_count"].dropna()
        print(f"\nData points per system:")
        print(f"  min:    {int(dp.min()):,}")
        print(f"  median: {int(dp.median()):,}")
        print(f"  max:    {int(dp.max()):,}")

    # --- System size ---
    # The systems_20241231.csv metadata does not include a rated capacity (kW/kWp)
    # column — size information is embedded in the per-system data files, not the
    # summary metadata.
    size_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ("size", "capacity", "kw", "kwp", "power")
    )]
    if size_cols:
        for col in size_cols:
            s = df[col].dropna()
            if len(s):
                print(f"\nSystem size — '{col}':")
                print(f"  min: {s.min()},  max: {s.max()},  median: {s.median()}")
    else:
        print(
            "\nNote: system rated capacity (kW) is not included in this metadata "
            "file — it is available in the individual system data CSVs."
        )

    # --- Date range ---
    for col in ("first_timestamp", "last_timestamp"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], format="%m/%d/%y %H:%M", errors="coerce")
            print(f"\n{col}: {parsed.min().date()} -> {parsed.max().date()}")

    # --- Sample rows ---
    print(f"\nFirst 5 rows:")
    display_cols = [c for c in ["system_id", "system_name", "location", "years",
                                "data_points_count"] if c in df.columns]
    print(df[display_cols].head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_or_download()
    summarise(df)


if __name__ == "__main__":
    main()
