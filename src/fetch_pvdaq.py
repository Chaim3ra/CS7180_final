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

import concurrent.futures
import io
import json
import os
import re

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

OUT_DIR         = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LEGACY_PATH     = os.path.join(OUT_DIR, "pvdaq_systems_metadata.csv")
RICH_PATH       = os.path.join(OUT_DIR, "pvdaq_systems_rich_metadata.csv")
JSON_DIR        = os.path.join(OUT_DIR, "pvdaq_system_jsons")
CANDIDATE_PATH  = os.path.join(OUT_DIR, "pvdaq_candidate_systems.csv")
SELECTED_PATH   = os.path.join(OUT_DIR, "pvdaq_selected_50.csv")

PVDATA_PREFIX   = "pvdaq/csv/pvdata/"
MAX_SIZE_MB     = 30.0    # skip systems whose best year exceeds this threshold
TARGET_COUNT    = 50      # number of systems to select
GRID_RES        = 0.1     # degrees — cell size for geographic diversity grid

# Continental US bounding box (excludes Hawaii, Alaska, Puerto Rico, etc.)
CONUS_LAT = (24.0, 50.0)
CONUS_LON = (-125.0, -66.0)

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
# Step 5: Filter candidate systems and save
# ---------------------------------------------------------------------------

def filter_candidates(df_rich: pd.DataFrame,
                      df_sensors: pd.DataFrame | None) -> pd.DataFrame:
    """
    Apply four filters to the rich metadata and return a candidate DataFrame.

    Filters
    -------
    1. qa_status == 'pass'
    2. dc_capacity_kW <= 20  (residential scale)
    3. years >= 1.0          (at least one full calendar year of data)
    4. CONUS lat/lon bounds  (excludes Hawaii, Alaska, Puerto Rico, etc.)

    Output columns
    --------------
    system_id, state, dc_capacity_kW, years, latitude, longitude,
    available_sensor_channels, has_weather_sensors
    """
    df = df_rich.copy()

    # ---- extract state from "City, ST" location strings ----
    df["state"] = df["site_location"].str.extract(r",\s*([A-Z]{2})\s*$")

    # ---- apply filters ----
    mask = (
        (df["qa_status"] == "pass")
        & df["dc_capacity_kW"].notna()
        & (df["dc_capacity_kW"] <= 20)
        & df["years"].notna()
        & (df["years"] >= 1.0)
        & df["latitude"].between(*CONUS_LAT)
        & df["longitude"].between(*CONUS_LON)
    )
    df_cand = df.loc[mask].copy()

    # ---- merge sensor classification ----
    # The per-system JSON files have empty "Other Instruments" sections for
    # virtually all systems, so keyword matching returns False everywhere.
    # Fall back to available_sensor_channels as the primary heuristic:
    #   - PVOutput crowd-sourced systems typically have 2 channels (AC power + energy)
    #   - NREL / research systems with dedicated pyranometers/weather stations
    #     have 10+ channels
    # This threshold may include systems with many inverters but no irradiance
    # sensors; those will be identified when time-series data is downloaded.
    SENSOR_CHANNEL_THRESHOLD = 10

    if df_sensors is not None and not df_sensors.empty:
        sensor_map = df_sensors.set_index("system_id")[
            ["has_irradiance", "has_weather"]
        ]
        df_cand = df_cand.join(sensor_map, on="system_id", how="left")
        json_flag = (
            df_cand["has_irradiance"].fillna(False)
            | df_cand["has_weather"].fillna(False)
        )
    else:
        json_flag = pd.Series(False, index=df_cand.index)

    channel_flag = (
        df_cand["available_sensor_channels"].fillna(0) >= SENSOR_CHANNEL_THRESHOLD
    )
    df_cand["has_weather_sensors"] = json_flag | channel_flag

    # ---- select and order output columns ----
    out_cols = [
        "system_id",
        "state",
        "dc_capacity_kW",
        "years",
        "latitude",
        "longitude",
        "available_sensor_channels",
        "has_weather_sensors",
    ]
    df_cand = (
        df_cand[out_cols]
        .sort_values(["has_weather_sensors", "years"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return df_cand


def print_and_save_candidates(df_cand: pd.DataFrame) -> None:
    """Print the filtered candidate list and save it to CSV."""
    print_separator("CANDIDATE SYSTEMS FOR DOWNLOAD")
    print(f"Filters applied:")
    print(f"  qa_status == 'pass'")
    print(f"  dc_capacity_kW <= 20  (residential scale)")
    print(f"  years >= 1.0          (at least 1 full year of data)")
    print(f"  CONUS lat {CONUS_LAT[0]}-{CONUS_LAT[1]}, "
          f"lon {CONUS_LON[0]} to {CONUS_LON[1]}")
    print(f"\nTotal candidates: {len(df_cand):,}")

    n_with_sensors = df_cand["has_weather_sensors"].sum()
    print(f"  With on-site weather/irradiance sensors: "
          f"{n_with_sensors:,} ({100*n_with_sensors/len(df_cand):.1f}%)")
    print(f"  Without dedicated sensors:               "
          f"{len(df_cand) - n_with_sensors:,}")

    # Summary stats on the candidate set
    print(f"\nCapacity (kW): "
          f"min={df_cand['dc_capacity_kW'].min():.2f}, "
          f"median={df_cand['dc_capacity_kW'].median():.2f}, "
          f"max={df_cand['dc_capacity_kW'].max():.2f}")
    print(f"Years of data: "
          f"min={df_cand['years'].min():.2f}, "
          f"median={df_cand['years'].median():.2f}, "
          f"max={df_cand['years'].max():.2f}")

    # State breakdown within candidates
    state_counts = (
        df_cand["state"]
        .value_counts(dropna=False)
        .rename_axis("state")
        .reset_index(name="candidates")
    )
    print(f"\nCandidates by state ({state_counts['state'].nunique()} states):")
    print(state_counts.to_string(index=False))

    # Full table — with_sensors first, then sorted by years desc
    print(f"\nFull candidate list (sorted: sensors first, then years desc):")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df_cand.to_string(index=True))
    pd.reset_option("display.max_rows")
    pd.reset_option("display.float_format")

    # Save
    df_cand.to_csv(CANDIDATE_PATH, index=False)
    print(f"\nSaved {len(df_cand):,} candidates -> {CANDIDATE_PATH}")
    print_separator()


# ---------------------------------------------------------------------------
# Step 6: Probe S3 for sensor-candidate systems and select top 50
# ---------------------------------------------------------------------------

# Regex for annual corrected file:  {sid}_ac_{YYYY}0101_{YYYY}1231_corrected.csv
_ANNUAL_RE = re.compile(r"_ac_(\d{4})0101_(\d{4})1231_corrected\.csv$")
# Regex for NREL daily file:        system_{sid}__date_{YYYY}_{MM}_{DD}.csv
_DAILY_RE  = re.compile(r"__date_(\d{4})_\d{2}_\d{2}\.csv$")


def _make_client():
    """Create a new unsigned S3 client. Must be called per-thread."""
    client = boto3.client("s3", region_name="us-west-2")
    client.meta.events.register("choose-signer.s3.*", disable_signing)
    return client


def probe_system_s3(sid: int) -> dict:
    """
    List S3 objects under pvdaq/csv/pvdata/system_id={sid}/ and determine
    the best downloadable year.

    Returns a dict with keys:
        system_id, has_data, file_format, best_year, size_mb, n_files, error
    """
    client  = _make_client()
    prefix  = f"{PVDATA_PREFIX}system_id={sid}/"
    base    = {"system_id": sid, "has_data": False, "file_format": None,
               "best_year": None, "size_mb": None, "n_files": 0, "error": None}

    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

        all_objects = []
        for page in pages:
            for obj in page.get("Contents", []):
                all_objects.append((obj["Key"], obj["Size"]))
    except Exception as exc:
        return {**base, "error": str(exc)}

    if not all_objects:
        return {**base, "error": "no files in prefix"}

    base["n_files"] = len(all_objects)

    # ----- Detect annual corrected files -----
    annual_years: dict[int, tuple[str, int]] = {}  # year -> (key, size_bytes)
    daily_year_bytes: dict[int, int] = {}           # year -> cumulative bytes

    for key, size in all_objects:
        fname = key.rsplit("/", 1)[-1]

        m_ann = _ANNUAL_RE.search(fname)
        if m_ann and m_ann.group(1) == m_ann.group(2):  # start_year == end_year
            year = int(m_ann.group(1))
            annual_years[year] = (key, size)
            continue

        m_day = _DAILY_RE.search(fname)
        if m_day:
            year = int(m_day.group(1))
            daily_year_bytes[year] = daily_year_bytes.get(year, 0) + size

    if annual_years:
        best_year = max(annual_years)
        best_key, best_bytes = annual_years[best_year]
        return {
            **base,
            "has_data":    True,
            "file_format": "annual",
            "best_year":   best_year,
            "size_mb":     round(best_bytes / 1_048_576, 2),
        }

    if daily_year_bytes:
        best_year = max(daily_year_bytes, key=daily_year_bytes.get)
        total_mb  = daily_year_bytes[best_year] / 1_048_576
        return {
            **base,
            "has_data":    True,
            "file_format": "daily",
            "best_year":   best_year,
            "size_mb":     round(total_mb, 2),
        }

    return {**base, "error": "no annual or daily files matched"}


def build_s3_inventory(df_cand: pd.DataFrame, max_workers: int = 20) -> pd.DataFrame:
    """
    Probe S3 for every sensor candidate system using a thread pool.
    Returns a DataFrame with one row per system.
    """
    sensor_sids = df_cand.loc[df_cand["has_weather_sensors"], "system_id"].tolist()
    n = len(sensor_sids)
    print(f"\n[6] Probing S3 inventory for {n} sensor-candidate systems ...")

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(probe_system_s3, sid): sid for sid in sensor_sids}
        for i, fut in enumerate(concurrent.futures.as_completed(future_map), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == n:
                within = sum(
                    1 for r in results
                    if r.get("has_data") and (r.get("size_mb") or 0) <= MAX_SIZE_MB
                )
                print(f"    {i}/{n} probed — {within} viable so far")

    df_inv = pd.DataFrame(results)
    # Attach lat/lon/state/years from candidates for the selection step
    meta = df_cand[["system_id", "state", "latitude", "longitude",
                    "years", "dc_capacity_kW"]].copy()
    df_inv = df_inv.merge(meta, on="system_id", how="left")
    return df_inv


def select_top_50(df_inv: pd.DataFrame) -> pd.DataFrame:
    """
    Select up to TARGET_COUNT systems from the S3 inventory that:
      - have at least one viable file (has_data=True)
      - have estimated year size <= MAX_SIZE_MB

    Selection prioritises geographic diversity using a GRID_RES-degree
    lat/lon grid, then fills remaining slots with the best remaining systems
    (sorted by years desc, most recent year desc, size asc).
    """
    # Viable: has data and within storage budget
    viable = df_inv[
        df_inv["has_data"].fillna(False)
        & (df_inv["size_mb"].fillna(float("inf")) <= MAX_SIZE_MB)
    ].copy()

    if viable.empty:
        print("  WARNING: no viable systems within size budget.")
        return viable

    # Assign grid cell
    viable["grid_lat"] = (viable["latitude"] / GRID_RES).round(0) * GRID_RES
    viable["grid_lon"] = (viable["longitude"] / GRID_RES).round(0) * GRID_RES
    viable["grid_cell"] = (
        viable["grid_lat"].map(lambda x: f"{x:.1f}")
        + "_"
        + viable["grid_lon"].map(lambda x: f"{x:.1f}")
    )

    # Sort: most years first, most recent best_year first, smallest file first
    viable = viable.sort_values(
        ["years", "best_year", "size_mb"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    used_cells:    set[str]        = set()
    used_ids:      set[int]        = set()

    # Pass 1 — one representative per grid cell
    for _, row in viable.iterrows():
        if len(selected_rows) >= TARGET_COUNT:
            break
        cell = row["grid_cell"]
        if cell not in used_cells:
            selected_rows.append(row)
            used_cells.add(cell)
            used_ids.add(row["system_id"])

    # Pass 2 — fill remaining slots with best leftover systems
    if len(selected_rows) < TARGET_COUNT:
        for _, row in viable.iterrows():
            if len(selected_rows) >= TARGET_COUNT:
                break
            if row["system_id"] not in used_ids:
                selected_rows.append(row)
                used_ids.add(row["system_id"])

    df_50 = pd.DataFrame(selected_rows).reset_index(drop=True)
    return df_50


def print_and_save_selected(df_50: pd.DataFrame) -> None:
    """Print the selected-50 list and save it to SELECTED_PATH."""
    print_separator("SELECTED 50 SYSTEMS")
    print(f"Systems selected: {len(df_50)}")

    # State breakdown
    state_counts = (
        df_50["state"].value_counts(dropna=False)
        .rename_axis("state").reset_index(name="count")
    )
    print(f"\nStates represented ({state_counts['state'].nunique()}):")
    print(state_counts.to_string(index=False))

    # Format breakdown
    if "file_format" in df_50.columns:
        fmt_counts = df_50["file_format"].value_counts()
        print(f"\nFile format: {dict(fmt_counts)}")

    # Size summary
    if "size_mb" in df_50.columns:
        mb = df_50["size_mb"].dropna()
        print(f"Year file size (MB): min={mb.min():.2f}, "
              f"median={mb.median():.2f}, max={mb.max():.2f}, "
              f"total={mb.sum():.1f}")

    # Full table
    display_cols = [c for c in [
        "system_id", "state", "latitude", "longitude",
        "dc_capacity_kW", "years", "file_format", "best_year", "size_mb",
    ] if c in df_50.columns]

    print(f"\nFull list (geographic diversity order):")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df_50[display_cols].to_string(index=True))
    pd.reset_option("display.max_rows")
    pd.reset_option("display.float_format")

    # Save
    save_cols = [c for c in df_50.columns if c not in ("grid_lat", "grid_lon", "grid_cell")]
    df_50[save_cols].to_csv(SELECTED_PATH, index=False)
    print(f"\nSaved {len(df_50)} systems -> {SELECTED_PATH}")
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

    df_cand = filter_candidates(df_rich, df_sensors)
    print_and_save_candidates(df_cand)

    # ----------------------------------------------------------------
    # Step 6: Select top 50 systems with geographic diversity
    # ----------------------------------------------------------------
    if os.path.exists(SELECTED_PATH):
        print(f"\n[6] Cache hit -> {SELECTED_PATH}")
        df_selected = pd.read_csv(SELECTED_PATH)
    else:
        df_inv      = build_s3_inventory(df_cand)
        df_selected = select_top_50(df_inv)

    print_and_save_selected(df_selected)


if __name__ == "__main__":
    main()
