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
import polars as pl
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


def s3_read_csv(bucket, key: str) -> pl.DataFrame:
    """Download a CSV from S3 into a Polars DataFrame without touching disk.

    Args:
        bucket: Boto3 Bucket resource.
        key: S3 object key.

    Returns:
        Polars DataFrame parsed from the CSV body.
    """
    body = bucket.Object(key).get()["Body"].read().decode("utf-8", errors="replace")
    return pl.read_csv(io.StringIO(body))


# ---------------------------------------------------------------------------
# Step 1: Download legacy snapshot via HTTPS
# ---------------------------------------------------------------------------

def fetch_legacy_metadata() -> pl.DataFrame:
    if os.path.exists(LEGACY_PATH):
        print(f"[1] Cache hit  -> {LEGACY_PATH}")
        return pl.read_csv(LEGACY_PATH)

    print(f"[1] Downloading {LEGACY_URL} ...")
    r = requests.get(LEGACY_URL, timeout=30)
    r.raise_for_status()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LEGACY_PATH, "wb") as f:
        f.write(r.content)
    print(f"    Saved {len(r.content):,} bytes -> {LEGACY_PATH}")
    return pl.read_csv(LEGACY_PATH)


# ---------------------------------------------------------------------------
# Step 2: Download rich metadata via boto3
# ---------------------------------------------------------------------------

def fetch_rich_metadata(bucket) -> pl.DataFrame:
    if os.path.exists(RICH_PATH):
        print(f"[2] Cache hit  -> {RICH_PATH}")
        return pl.read_csv(RICH_PATH)

    print(f"[2] Downloading s3://{BUCKET_NAME}/{RICH_S3_KEY} via boto3 ...")
    df = s3_read_csv(bucket, RICH_S3_KEY)
    # Drop unnamed trailing columns from spreadsheet artifacts
    df = df.select([c for c in df.columns if not c.startswith("Unnamed")])
    os.makedirs(OUT_DIR, exist_ok=True)
    df.write_csv(RICH_PATH)
    print(f"    Saved {len(df):,} rows -> {RICH_PATH}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Download and parse per-system metadata JSONs
# ---------------------------------------------------------------------------

def fetch_system_jsons(bucket) -> list[dict]:
    """Paginate system_metadata/ in S3 and download all per-system JSON files.

    Results are cached on disk in ``data/raw/pvdaq_system_jsons/``.

    Args:
        bucket: Boto3 Bucket resource.

    Returns:
        List of parsed JSON dicts, one per system.
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


def classify_sensors(records: list[dict]) -> pl.DataFrame:
    """Build a per-system sensor classification DataFrame from JSON records.

    Checks ``Other Instruments`` and ``Meters`` sections for irradiance and
    weather sensors.

    Args:
        records: List of parsed per-system JSON dicts.

    Returns:
        Polars DataFrame with columns:
        ``system_id``, ``has_irradiance``, ``has_weather``, ``other_instruments``.
    """
    rows = []
    for rec in records:
        sys_info = rec.get("System", {})
        other    = rec.get("Other Instruments", {})
        meters   = rec.get("Meters", {})
        metrics  = rec.get("Metrics", {})

        all_instruments = {**other, **meters}

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
    return pl.from_dicts(rows) if rows else pl.DataFrame(
        {"system_id": [], "has_irradiance": [], "has_weather": [], "other_instruments": []}
    )


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


def summarise(
    df_legacy: pl.DataFrame,
    df_rich: pl.DataFrame,
    df_sensors: pl.DataFrame | None,
) -> None:
    """Print a comprehensive analysis of PVDAQ system metadata.

    Args:
        df_legacy: Legacy snapshot DataFrame (systems_20241231.csv).
        df_rich: Rich metadata DataFrame (systems_20250729.csv).
        df_sensors: Per-system sensor classification, or ``None``.
    """
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
    # 2. State breakdown
    # ------------------------------------------------------------------
    print_separator("SYSTEMS BY STATE")
    df_rich = df_rich.with_columns(
        pl.col("site_location")
        .str.extract(r",\s*([A-Z]{2})\s*$", group_index=1)
        .alias("state")
    )
    state_counts = (
        df_rich["state"]
        .value_counts(sort=True)
        .rename({"state": "state", "count": "systems"})
    )
    n_states = df_rich["state"].n_unique()
    print(f"{len(df_rich):,} systems across {n_states} states/territories:\n")
    print(state_counts)

    # ------------------------------------------------------------------
    # 3. System size distribution
    # ------------------------------------------------------------------
    print_separator("SYSTEM SIZE  (dc_capacity_kW)")
    if "dc_capacity_kW" in df_rich.columns:
        kw = df_rich["dc_capacity_kW"].drop_nulls()
        print(f"Systems with known capacity: {len(kw):,} / {len(df_rich):,}")
        print(f"  min    : {kw.min():.3f} kW")
        print(f"  p25    : {kw.quantile(0.25):.3f} kW")
        print(f"  median : {kw.median():.3f} kW")
        print(f"  mean   : {kw.mean():.3f} kW")
        print(f"  p75    : {kw.quantile(0.75):.3f} kW")
        print(f"  max    : {kw.max():.3f} kW")

        df_rich = df_rich.with_columns(
            pl.col("dc_capacity_kW")
            .cut(
                breaks=[10, 100, 1000],
                labels=[
                    "<10 kW (residential)",
                    "10-100 kW (small commercial)",
                    "100-1000 kW (commercial)",
                    ">1 MW (utility)",
                ],
            )
            .alias("capacity_tier")
        )
        tier_counts = (
            df_rich.group_by("capacity_tier")
            .len()
            .sort("capacity_tier")
        )
        print("\nCapacity tiers:")
        for row in tier_counts.rows(named=True):
            print(f"  {row['capacity_tier']}: {row['len']:,}")
    else:
        print("  dc_capacity_kW not present in this snapshot.")

    # ------------------------------------------------------------------
    # 4. Date range coverage
    # ------------------------------------------------------------------
    print_separator("DATA COVERAGE")
    for col in ("first_timestamp", "last_timestamp"):
        if col in df_rich.columns:
            parsed = df_rich[col].str.to_datetime(
                format="%m/%d/%Y %H:%M", strict=False
            )
            print(f"  {col}: {parsed.min()} -> {parsed.max()}")
    if "years" in df_rich.columns:
        yr = df_rich["years"].drop_nulls()
        print(f"  Years of data per system: "
              f"min={yr.min():.2f}, median={yr.median():.2f}, max={yr.max():.2f}")

    if "tracking" in df_rich.columns:
        print_separator("TRACKING TYPE")
        print(df_rich["tracking"].value_counts(sort=True))

    if "type" in df_rich.columns:
        print_separator("MOUNT TYPE")
        print(df_rich["type"].value_counts(sort=True))

    if "qa_status" in df_rich.columns:
        print_separator("QA STATUS")
        print(df_rich["qa_status"].value_counts(sort=True))

    # ------------------------------------------------------------------
    # 5. On-site irradiance / weather sensors
    # ------------------------------------------------------------------
    print_separator("ON-SITE IRRADIANCE / WEATHER SENSORS")
    if df_sensors is not None and not df_sensors.is_empty():
        n_total  = len(df_sensors)
        n_irr    = df_sensors["has_irradiance"].sum()
        n_wth    = df_sensors["has_weather"].sum()
        n_either = (
            df_sensors
            .select(
                (pl.col("has_irradiance") | pl.col("has_weather")).alias("either")
            )["either"]
            .sum()
        )
        print(f"Systems with per-JSON metadata: {n_total:,}")
        print(f"  Has irradiance sensor : {n_irr:,}  ({100*n_irr/n_total:.1f}%)")
        print(f"  Has weather sensor    : {n_wth:,}  ({100*n_wth/n_total:.1f}%)")
        print(f"  Has either            : {n_either:,}  ({100*n_either/n_total:.1f}%)")

        if "available_sensor_channels" in df_rich.columns:
            ch = df_rich["available_sensor_channels"].drop_nulls()
            print(f"\navailable_sensor_channels across all {len(ch):,} systems:")
            print(f"  min={ch.min():.0f}, median={ch.median():.0f}, max={ch.max():.0f}")
            threshold = 10
            n_likely = (ch >= threshold).sum()
            print(f"  Systems with >= {threshold} channels (likely have dedicated sensors): "
                  f"{n_likely:,} ({100*n_likely/len(ch):.1f}%)")
    else:
        print("  No per-system JSON metadata available for sensor analysis.")
        if "available_sensor_channels" in df_rich.columns:
            ch = df_rich["available_sensor_channels"].drop_nulls()
            threshold = 10
            n_likely = (ch >= threshold).sum()
            print(f"\n  Proxy via available_sensor_channels >= {threshold}: "
                  f"{n_likely:,} / {len(ch):,} systems ({100*n_likely/len(ch):.1f}%)")

    print_separator()


# ---------------------------------------------------------------------------
# Step 5: Filter candidate systems and save
# ---------------------------------------------------------------------------

def filter_candidates(
    df_rich: pl.DataFrame,
    df_sensors: pl.DataFrame | None,
) -> pl.DataFrame:
    """Apply four filters to the rich metadata and return a candidate DataFrame.

    Filters
    -------
    1. qa_status == 'pass'
    2. dc_capacity_kW <= 20  (residential scale)
    3. years >= 1.0          (at least one full calendar year of data)
    4. CONUS lat/lon bounds  (excludes Hawaii, Alaska, Puerto Rico, etc.)

    Args:
        df_rich: Rich metadata Polars DataFrame.
        df_sensors: Per-system sensor classification, or ``None``.

    Returns:
        Filtered candidate DataFrame with columns:
        ``system_id``, ``state``, ``dc_capacity_kW``, ``years``,
        ``latitude``, ``longitude``, ``available_sensor_channels``,
        ``has_weather_sensors``.
    """
    df = df_rich.with_columns(
        pl.col("site_location")
        .str.extract(r",\s*([A-Z]{2})\s*$", group_index=1)
        .alias("state")
    )

    df_cand = df.filter(
        (pl.col("qa_status") == "pass")
        & pl.col("dc_capacity_kW").is_not_null()
        & (pl.col("dc_capacity_kW") <= 20)
        & pl.col("years").is_not_null()
        & (pl.col("years") >= 1.0)
        & pl.col("latitude").is_between(CONUS_LAT[0], CONUS_LAT[1])
        & pl.col("longitude").is_between(CONUS_LON[0], CONUS_LON[1])
    )

    SENSOR_CHANNEL_THRESHOLD = 10

    if df_sensors is not None and not df_sensors.is_empty():
        df_cand = df_cand.join(
            df_sensors.select(["system_id", "has_irradiance", "has_weather"]),
            on="system_id",
            how="left",
        )
        json_flag = (
            pl.col("has_irradiance").fill_null(False)
            | pl.col("has_weather").fill_null(False)
        )
    else:
        df_cand = df_cand.with_columns([
            pl.lit(False).alias("has_irradiance"),
            pl.lit(False).alias("has_weather"),
        ])
        json_flag = pl.lit(False)

    df_cand = df_cand.with_columns(
        (
            json_flag
            | (pl.col("available_sensor_channels").fill_null(0) >= SENSOR_CHANNEL_THRESHOLD)
        ).alias("has_weather_sensors")
    )

    out_cols = [
        "system_id", "state", "dc_capacity_kW", "years",
        "latitude", "longitude", "available_sensor_channels", "has_weather_sensors",
    ]
    return (
        df_cand
        .select(out_cols)
        .sort(["has_weather_sensors", "years"], descending=[True, True])
    )


def print_and_save_candidates(df_cand: pl.DataFrame) -> None:
    """Print the filtered candidate list and save it to CSV.

    Args:
        df_cand: Filtered candidate Polars DataFrame.
    """
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

    cap = df_cand["dc_capacity_kW"]
    print(f"\nCapacity (kW): "
          f"min={cap.min():.2f}, median={cap.median():.2f}, max={cap.max():.2f}")
    yrs = df_cand["years"]
    print(f"Years of data: "
          f"min={yrs.min():.2f}, median={yrs.median():.2f}, max={yrs.max():.2f}")

    state_counts = df_cand["state"].value_counts(sort=True)
    print(f"\nCandidates by state ({df_cand['state'].n_unique()} states):")
    print(state_counts)

    print(f"\nFull candidate list (sorted: sensors first, then years desc):")
    with pl.Config(tbl_rows=len(df_cand), tbl_width_chars=120):
        print(df_cand)

    df_cand.write_csv(CANDIDATE_PATH)
    print(f"\nSaved {len(df_cand):,} candidates -> {CANDIDATE_PATH}")
    print_separator()


# ---------------------------------------------------------------------------
# Step 6: Probe S3 for sensor-candidate systems and select top 50
# ---------------------------------------------------------------------------

_ANNUAL_RE = re.compile(r"_ac_(\d{4})0101_(\d{4})1231_corrected\.csv$")
_DAILY_RE  = re.compile(r"__date_(\d{4})_\d{2}_\d{2}\.csv$")


def _make_client():
    """Create a new unsigned S3 client (one per thread)."""
    client = boto3.client("s3", region_name="us-west-2")
    client.meta.events.register("choose-signer.s3.*", disable_signing)
    return client


def probe_system_s3(sid: int) -> dict:
    """List S3 objects under ``pvdaq/csv/pvdata/system_id={sid}/``.

    Args:
        sid: PVDAQ system ID.

    Returns:
        Dict with keys: ``system_id``, ``has_data``, ``file_format``,
        ``best_year``, ``size_mb``, ``n_files``, ``error``.
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

    annual_years: dict[int, tuple[str, int]] = {}
    daily_year_bytes: dict[int, int] = {}

    for key, size in all_objects:
        fname = key.rsplit("/", 1)[-1]
        m_ann = _ANNUAL_RE.search(fname)
        if m_ann and m_ann.group(1) == m_ann.group(2):
            year = int(m_ann.group(1))
            annual_years[year] = (key, size)
            continue
        m_day = _DAILY_RE.search(fname)
        if m_day:
            year = int(m_day.group(1))
            daily_year_bytes[year] = daily_year_bytes.get(year, 0) + size

    if annual_years:
        best_year = max(annual_years)
        _, best_bytes = annual_years[best_year]
        return {
            **base,
            "has_data":    True,
            "file_format": "annual",
            "best_year":   best_year,
            "size_mb":     round(best_bytes / 1_048_576, 2),
        }

    if daily_year_bytes:
        best_year = max(daily_year_bytes, key=daily_year_bytes.get)
        return {
            **base,
            "has_data":    True,
            "file_format": "daily",
            "best_year":   best_year,
            "size_mb":     round(daily_year_bytes[best_year] / 1_048_576, 2),
        }

    return {**base, "error": "no annual or daily files matched"}


def build_s3_inventory(df_cand: pl.DataFrame, max_workers: int = 20) -> pl.DataFrame:
    """Probe S3 for every sensor-candidate system using a thread pool.

    Args:
        df_cand: Candidate systems Polars DataFrame.
        max_workers: Thread pool size.

    Returns:
        Polars DataFrame with one row per system and S3 inventory info.
    """
    sensor_sids = (
        df_cand
        .filter(pl.col("has_weather_sensors"))["system_id"]
        .to_list()
    )
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

    df_inv = pl.from_dicts(results)
    meta = df_cand.select(
        ["system_id", "state", "latitude", "longitude", "years", "dc_capacity_kW"]
    )
    return df_inv.join(meta, on="system_id", how="left")


def select_top_50(df_inv: pl.DataFrame) -> pl.DataFrame:
    """Select up to ``TARGET_COUNT`` systems with geographic diversity.

    Filters to viable systems (has data and within ``MAX_SIZE_MB``), assigns
    each to a ``GRID_RES``-degree lat/lon cell, then greedily selects one
    representative per cell before filling remaining slots with the best
    unselected systems.

    Args:
        df_inv: S3 inventory Polars DataFrame from :func:`build_s3_inventory`.

    Returns:
        Polars DataFrame of selected systems.
    """
    viable = df_inv.filter(
        pl.col("has_data").fill_null(False)
        & (pl.col("size_mb").fill_null(float("inf")) <= MAX_SIZE_MB)
    )

    if viable.is_empty():
        print("  WARNING: no viable systems within size budget.")
        return viable

    viable = (
        viable
        .with_columns([
            ((pl.col("latitude")  / GRID_RES).round(0) * GRID_RES).alias("grid_lat"),
            ((pl.col("longitude") / GRID_RES).round(0) * GRID_RES).alias("grid_lon"),
        ])
        .with_columns(
            (
                pl.col("grid_lat").map_elements(lambda x: f"{x:.1f}", return_dtype=pl.String)
                + pl.lit("_")
                + pl.col("grid_lon").map_elements(lambda x: f"{x:.1f}", return_dtype=pl.String)
            ).alias("grid_cell")
        )
        .sort(["years", "best_year", "size_mb"], descending=[True, True, False])
    )

    selected_rows: list[dict] = []
    used_cells:    set[str]   = set()
    used_ids:      set[int]   = set()

    for row in viable.rows(named=True):
        if len(selected_rows) >= TARGET_COUNT:
            break
        cell = row["grid_cell"]
        if cell not in used_cells:
            selected_rows.append(row)
            used_cells.add(cell)
            used_ids.add(row["system_id"])

    if len(selected_rows) < TARGET_COUNT:
        for row in viable.rows(named=True):
            if len(selected_rows) >= TARGET_COUNT:
                break
            if row["system_id"] not in used_ids:
                selected_rows.append(row)
                used_ids.add(row["system_id"])

    return pl.from_dicts(selected_rows)


def print_and_save_selected(df_50: pl.DataFrame) -> None:
    """Print the selected-50 list and save it to ``SELECTED_PATH``.

    Args:
        df_50: Selected systems Polars DataFrame.
    """
    print_separator("SELECTED 50 SYSTEMS")
    print(f"Systems selected: {len(df_50)}")

    if "state" in df_50.columns:
        state_counts = df_50["state"].value_counts(sort=True)
        print(f"\nStates represented ({df_50['state'].n_unique()}):")
        print(state_counts)

    if "file_format" in df_50.columns:
        fmt_counts = df_50["file_format"].value_counts(sort=True)
        print(f"\nFile format:")
        print(fmt_counts)

    if "size_mb" in df_50.columns:
        mb = df_50["size_mb"].drop_nulls()
        print(f"Year file size (MB): min={mb.min():.2f}, "
              f"median={mb.median():.2f}, max={mb.max():.2f}, "
              f"total={mb.sum():.1f}")

    display_cols = [c for c in [
        "system_id", "state", "latitude", "longitude",
        "dc_capacity_kW", "years", "file_format", "best_year", "size_mb",
    ] if c in df_50.columns]

    print(f"\nFull list (geographic diversity order):")
    with pl.Config(tbl_rows=len(df_50), tbl_width_chars=140):
        print(df_50.select(display_cols))

    save_cols = [
        c for c in df_50.columns
        if c not in ("grid_lat", "grid_lon", "grid_cell")
    ]
    df_50.select(save_cols).write_csv(SELECTED_PATH)
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

    if os.path.exists(SELECTED_PATH):
        print(f"\n[6] Cache hit -> {SELECTED_PATH}")
        df_selected = pl.read_csv(SELECTED_PATH)
    else:
        df_inv      = build_s3_inventory(df_cand)
        df_selected = select_top_50(df_inv)

    print_and_save_selected(df_selected)


if __name__ == "__main__":
    main()
