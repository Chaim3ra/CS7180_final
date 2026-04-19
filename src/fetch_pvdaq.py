"""
fetch_pvdaq.py
--------------
Explore PVDAQ system metadata from the DOE OEDI S3 data lake.
Uses the pvdaq_access package's boto3 pattern for authenticated-free S3 access.

Steps
-----
1. Download systems_20241231.csv (the snapshot URL given in task spec) via HTTPS
   -> data/raw/pvdaq_systems_metadata.csv  (skip if already cached)
2. Download systems_20250729.csv via boto3 (richer: has dc_capacity_kW,
   available_sensor_channels, tracking type, QA status, climate zone)
   -> data/raw/pvdaq_systems_rich_metadata.csv  (skip if already cached)
3. Paginate the system_metadata/ prefix in S3 and download all per-system
   JSON files (each ~2 KB, metadata only) to count on-site irradiance and
   weather instruments.
4. Print a comprehensive summary:
   - Total systems and column inventory
   - State breakdown
   - System size distribution (dc_capacity_kW)
   - Data coverage date range
   - On-site irradiance / weather sensor count

Usage:
    python src/fetch_pvdaq.py

No API key required — the OEDI bucket is public.
"""

import io
import json
import os

import boto3
import pandas as pd
import requests
from botocore.handlers import disable_signing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKET_NAME = "oedi-data-lake"

LEGACY_URL  = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20241231.csv"
RICH_S3_KEY = "pvdaq/csv/systems_20250729.csv"
META_PREFIX = "pvdaq/csv/system_metadata/"

OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LEGACY_PATH  = os.path.join(OUT_DIR, "pvdaq_systems_metadata.csv")
RICH_PATH    = os.path.join(OUT_DIR, "pvdaq_systems_rich_metadata.csv")
JSON_DIR     = os.path.join(OUT_DIR, "pvdaq_system_jsons")

# Keywords that indicate a dedicated on-site sensor (not just an inverter channel)
IRRADIANCE_KEYWORDS = {
    "poa", "ghi", "dni", "dhi", "irradiance", "pyranometer",
}
WEATHER_KEYWORDS = {
    "wind", "anemometer", "temperature", "humidity", "pressure",
    "weather", "ambient", "air_temp",
}
SENSOR_KEYWORDS = IRRADIANCE_KEYWORDS | WEATHER_KEYWORDS


# ---------------------------------------------------------------------------
# S3 helper (pvdaq_access pattern — unsigned public bucket)
# ---------------------------------------------------------------------------

def get_bucket():
    s3 = boto3.resource("s3")
    s3.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)
    return s3.Bucket(BUCKET_NAME)


def s3_read_csv(bucket, key: str) -> pd.DataFrame:
    """Download a CSV from S3 into a DataFrame without touching disk."""
    body = bucket.Object(key).get()["Body"].read().decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(body))


# ---------------------------------------------------------------------------
# Step 1: Download legacy snapshot via HTTPS
# ---------------------------------------------------------------------------

def fetch_legacy_metadata() -> pd.DataFrame:
    if os.path.exists(LEGACY_PATH):
        print(f"[1] Cache hit  -> {LEGACY_PATH}")
        return pd.read_csv(LEGACY_PATH)

    print(f"[1] Downloading {LEGACY_URL} ...")
    r = requests.get(LEGACY_URL, timeout=30)
    r.raise_for_status()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LEGACY_PATH, "wb") as f:
        f.write(r.content)
    print(f"    Saved {len(r.content):,} bytes -> {LEGACY_PATH}")
    return pd.read_csv(LEGACY_PATH)


# ---------------------------------------------------------------------------
# Step 2: Download rich metadata via boto3 (pvdaq_access pattern)
# ---------------------------------------------------------------------------

def fetch_rich_metadata(bucket) -> pd.DataFrame:
    if os.path.exists(RICH_PATH):
        print(f"[2] Cache hit  -> {RICH_PATH}")
        df = pd.read_csv(RICH_PATH)
    else:
        print(f"[2] Downloading s3://{BUCKET_NAME}/{RICH_S3_KEY} via boto3 ...")
        df = s3_read_csv(bucket, RICH_S3_KEY)
        # Drop unnamed trailing columns from spreadsheet artifacts
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        os.makedirs(OUT_DIR, exist_ok=True)
        df.to_csv(RICH_PATH, index=False)
        print(f"    Saved {len(df):,} rows -> {RICH_PATH}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Download and parse per-system metadata JSONs
# ---------------------------------------------------------------------------

def fetch_system_jsons(bucket) -> list[dict]:
    """
    Paginate system_metadata/ in S3 and download all *_system_metadata.json
    files (each ~2 KB). Results are cached on disk in data/raw/pvdaq_system_jsons/.
    Returns a list of parsed JSON dicts.
    """
    os.makedirs(JSON_DIR, exist_ok=True)
    paginator = bucket.meta.client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=META_PREFIX)

    json_keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("_system_metadata.json"):
                json_keys.append(obj["Key"])

    print(f"[3] Found {len(json_keys):,} system metadata JSON files in S3.")

    records = []
    downloaded = 0
    for key in json_keys:
        fname = os.path.basename(key)
        local = os.path.join(JSON_DIR, fname)
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            body = bucket.Object(key).get()["Body"].read()
            with open(local, "wb") as f:
                f.write(body)
            data = json.loads(body)
            downloaded += 1
        records.append(data)

    cached = len(json_keys) - downloaded
    print(f"    Downloaded {downloaded:,} new | used {cached:,} cached")
    return records


def has_sensor(instruments: dict, keywords: set) -> bool:
    """Return True if any instrument name/type matches any keyword."""
    for inst_info in instruments.values():
        if not isinstance(inst_info, dict):
            continue
        searchable = " ".join([
            str(inst_info.get("name", "")),
            str(inst_info.get("type", "")),
            str(inst_info.get("manufacturer", "")),
            str(inst_info.get("model", "")),
        ]).lower()
        if any(kw in searchable for kw in keywords):
            return True
    return False


def classify_sensors(records: list[dict]) -> pd.DataFrame:
    """
    Build a per-system sensor classification DataFrame from the JSON records.
    Checks 'Other Instruments' and 'Meters' sections for irradiance/weather sensors.
    """
    rows = []
    for rec in records:
        sys_info  = rec.get("System", {})
        other     = rec.get("Other Instruments", {})
        meters    = rec.get("Meters", {})
        metrics   = rec.get("Metrics", {})

        # Combine all non-inverter/non-module instrument dicts
        all_instruments = {**other, **meters}

        # Also scan metric sensor names for irradiance channels
        metric_names = " ".join(
            str(m.get("sensor_name", "")) for m in metrics.values()
        ).lower()
        has_irr_metric = any(kw in metric_names for kw in IRRADIANCE_KEYWORDS)
        has_wth_metric = any(kw in metric_names for kw in WEATHER_KEYWORDS)

        rows.append({
            "system_id":         sys_info.get("system_id"),
            "has_irradiance":    has_sensor(all_instruments, IRRADIANCE_KEYWORDS) or has_irr_metric,
            "has_weather":       has_sensor(all_instruments, WEATHER_KEYWORDS)    or has_wth_metric,
            "other_instruments": len(other),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: Print analysis
# ---------------------------------------------------------------------------

def print_separator(title=""):
    width = 64
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'=' * pad} {title} {'=' * pad}")
    else:
        print("=" * width)


def summarise(df_legacy: pd.DataFrame, df_rich: pd.DataFrame,
              df_sensors: pd.DataFrame | None) -> None:

    # ------------------------------------------------------------------
    # 1. Overview
    # ------------------------------------------------------------------
    print_separator("OVERVIEW")
    print(f"Snapshot file (systems_20241231.csv): {len(df_legacy):,} systems")
    print(f"Rich file     (systems_20250729.csv): {len(df_rich):,} systems")
    print(f"\nLegacy columns ({len(df_legacy.columns)}):")
    for c in df_legacy.columns:
        print(f"  {c}")
    print(f"\nRich columns ({len(df_rich.columns)}):")
    for c in df_rich.columns:
        print(f"  {c}")

    # ------------------------------------------------------------------
    # 2. State breakdown (rich file — more complete)
    # ------------------------------------------------------------------
    print_separator("SYSTEMS BY STATE")
    df_rich = df_rich.copy()
    df_rich["state"] = df_rich["site_location"].str.extract(r",\s*([A-Z]{2})\s*$")
    state_counts = (
        df_rich["state"]
        .value_counts()
        .rename_axis("state")
        .reset_index(name="systems")
    )
    n_states = state_counts["state"].nunique()
    print(f"{len(df_rich):,} systems across {n_states} states/territories:\n")
    print(state_counts.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. System size distribution (dc_capacity_kW)
    # ------------------------------------------------------------------
    print_separator("SYSTEM SIZE  (dc_capacity_kW)")
    if "dc_capacity_kW" in df_rich.columns:
        kw = df_rich["dc_capacity_kW"].dropna()
        print(f"Systems with known capacity: {len(kw):,} / {len(df_rich):,}")
        print(f"  min    : {kw.min():.3f} kW")
        print(f"  p25    : {kw.quantile(0.25):.3f} kW")
        print(f"  median : {kw.median():.3f} kW")
        print(f"  mean   : {kw.mean():.3f} kW")
        print(f"  p75    : {kw.quantile(0.75):.3f} kW")
        print(f"  max    : {kw.max():.3f} kW")

        # Capacity tiers
        bins   = [0, 10, 100, 1000, float("inf")]
        labels = ["<10 kW (residential)", "10-100 kW (small commercial)",
                  "100-1000 kW (commercial)", ">1 MW (utility)"]
        df_rich["capacity_tier"] = pd.cut(kw, bins=bins, labels=labels)
        tier_counts = df_rich["capacity_tier"].value_counts().sort_index()
        print("\nCapacity tiers:")
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count:,}")
    else:
        print("  dc_capacity_kW not present in this snapshot.")

    # ------------------------------------------------------------------
    # 4. Date range coverage
    # ------------------------------------------------------------------
    print_separator("DATA COVERAGE")
    for col in ("first_timestamp", "last_timestamp"):
        col_name = col if col in df_rich.columns else None
        if col_name:
            parsed = pd.to_datetime(df_rich[col_name], format="%m/%d/%Y %H:%M",
                                    errors="coerce")
            print(f"  {col}: {parsed.min().date()} -> {parsed.max().date()}")
    if "years" in df_rich.columns:
        yr = df_rich["years"].dropna()
        print(f"  Years of data per system: "
              f"min={yr.min():.2f}, median={yr.median():.2f}, max={yr.max():.2f}")

    # Tracking type breakdown
    if "tracking" in df_rich.columns:
        print_separator("TRACKING TYPE")
        print(df_rich["tracking"].value_counts().rename_axis("tracking").reset_index(
            name="systems").to_string(index=False))

    # Mount type breakdown
    if "type" in df_rich.columns:
        print_separator("MOUNT TYPE")
        print(df_rich["type"].value_counts().rename_axis("type").reset_index(
            name="systems").to_string(index=False))

    # QA status
    if "qa_status" in df_rich.columns:
        print_separator("QA STATUS")
        print(df_rich["qa_status"].value_counts().rename_axis("qa_status").reset_index(
            name="systems").to_string(index=False))

    # ------------------------------------------------------------------
    # 5. On-site irradiance / weather sensors
    # ------------------------------------------------------------------
    print_separator("ON-SITE IRRADIANCE / WEATHER SENSORS")
    if df_sensors is not None and not df_sensors.empty:
        n_total  = len(df_sensors)
        n_irr    = df_sensors["has_irradiance"].sum()
        n_wth    = df_sensors["has_weather"].sum()
        n_either = (df_sensors["has_irradiance"] | df_sensors["has_weather"]).sum()
        print(f"Systems with per-JSON metadata: {n_total:,}")
        print(f"  Has irradiance sensor : {n_irr:,}  ({100*n_irr/n_total:.1f}%)")
        print(f"  Has weather sensor    : {n_wth:,}  ({100*n_wth/n_total:.1f}%)")
        print(f"  Has either            : {n_either:,}  ({100*n_either/n_total:.1f}%)")

        # Sensor channel proxy for the broader dataset
        if "available_sensor_channels" in df_rich.columns:
            ch = df_rich["available_sensor_channels"].dropna()
            print(f"\navailable_sensor_channels across all {len(ch):,} systems:")
            print(f"  min={ch.min():.0f}, median={ch.median():.0f}, max={ch.max():.0f}")
            threshold = 10   # heuristic: >10 channels usually means dedicated sensors
            n_likely  = (ch >= threshold).sum()
            print(f"  Systems with >= {threshold} channels (likely have dedicated sensors): "
                  f"{n_likely:,} ({100*n_likely/len(ch):.1f}%)")
    else:
        print("  No per-system JSON metadata available for sensor analysis.")
        if "available_sensor_channels" in df_rich.columns:
            ch = df_rich["available_sensor_channels"].dropna()
            threshold = 10
            n_likely  = (ch >= threshold).sum()
            print(f"\n  Proxy via available_sensor_channels >= {threshold}: "
                  f"{n_likely:,} / {len(ch):,} systems ({100*n_likely/len(ch):.1f}%)")

    print_separator()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bucket = get_bucket()

    df_legacy  = fetch_legacy_metadata()
    df_rich    = fetch_rich_metadata(bucket)
    json_recs  = fetch_system_jsons(bucket)
    df_sensors = classify_sensors(json_recs) if json_recs else None

    summarise(df_legacy, df_rich, df_sensors)


if __name__ == "__main__":
    main()
