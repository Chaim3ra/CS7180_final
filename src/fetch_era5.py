"""Fetch ERA5 hourly reanalysis weather data from Copernicus CDS.

Downloads one NetCDF file per (location, year) and derives the six weather
columns used by the solar forecasting model:

    GHI_W_m2        surface solar radiation downwards    (ssrd / 3600)
    DNI_W_m2        direct normal irradiance             (fdir / cos(SZA) / 3600)
    DHI_W_m2        diffuse horizontal irradiance        ((ssrd - fdir) / 3600)
    Temp_C          2m air temperature                   (t2m - 273.15)
    WindSpeed_m_s   10m wind speed                       (sqrt(u10^2 + v10^2))
    RelHumidity_pct relative humidity                    (Magnus formula from d2m, t2m)

ERA5 radiation variables are accumulated over each 1-hour step (J/m²).
Dividing by 3600 converts to mean W/m² for that hour.

DNI derivation requires the solar zenith angle (SZA), computed via pvlib using
the target lat/lon and UTC timestamps.  At SZA >= 90 deg (night) DNI = 0.

Outputs
-------
data/raw/era5_{label}_{year}.nc   NetCDF4, raw ERA5 variables
data/raw/era5_{label}_{year}.csv  CSV with derived model columns + datetime

Caching: skips if the CSV already exists (pass --force to reprocess).

Setup
-----
Requires a free CDS account: https://cds.climate.copernicus.eu

1. Register at the URL above.
2. Accept the ERA5 licence on the dataset page.
3. Copy your API key from https://cds.climate.copernicus.eu/profile
4. Add to .env:
       CDS_API_KEY=your-api-key-here
       CDS_API_URL=https://cds.climate.copernicus.eu/api   # default if omitted

Usage
-----
    python src/fetch_era5.py                         # fetch default locations/years
    python src/fetch_era5.py --force                 # re-download even if cached
    python src/fetch_era5.py --systems pvdaq.csv     # read lat/lon/label from CSV
    python src/fetch_era5.py --location austin --year 2018
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# -- Repo root and .env -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for raw in env_path.read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

DATA_RAW = ROOT / "data" / "raw"

# -- Dependency checks --------------------------------------------------------

def _require(package: str, install: str = "") -> None:
    """Raise ImportError with pip hint if a package is missing."""
    try:
        __import__(package)
    except ImportError:
        hint = f"pip install {install or package}"
        raise ImportError(
            f"'{package}' is required but not installed.\n"
            f"Install with: {hint}"
        ) from None


# -- Constants ----------------------------------------------------------------

CDS_DATASET = "reanalysis-era5-single-levels"

# ERA5 short names -> CDS API variable names
ERA5_VARIABLES = [
    "surface_solar_radiation_downwards",          # ssrd -> GHI
    "total_sky_direct_solar_radiation_at_surface", # fdir -> used for DNI + DHI
    "2m_temperature",                              # t2m  -> Temp_C
    "2m_dewpoint_temperature",                     # d2m  -> RelHumidity_pct
    "10m_u_component_of_wind",                     # u10  -> WindSpeed_m_s (component)
    "10m_v_component_of_wind",                     # v10  -> WindSpeed_m_s (component)
]

# Default fetch targets: representative locations covering PVDAQ + Pecan Street regions.
# Each entry: (label, lat, lon, years)
# - label must be filesystem-safe (used in filename)
# - Actual PVDAQ system coordinates will be read from src/fetch_pvdaq.py output CSV
DEFAULT_LOCATIONS: list[tuple[str, float, float, list[int]]] = [
    # Pecan Street training locations (matching v1 NASA POWER fetches)
    ("austin",     30.2672, -97.7431,  [2018]),
    ("california", 37.3382, -121.8863, [2014, 2015]),
    ("newyork",    40.7128, -74.0060,  [2019]),
    # Additional PVDAQ region representatives (to be expanded from pvdaq.csv)
    ("albuquerque", 35.0853, -106.6056, [2018, 2019, 2020]),
    ("phoenix",     33.4484, -112.0740, [2018, 2019, 2020]),
    ("denver",      39.7392, -104.9903, [2018, 2019, 2020]),
]


# -- Helpers ------------------------------------------------------------------

def _cds_client(url: Optional[str] = None, key: Optional[str] = None):
    """Build and return a cdsapi.Client using env vars or provided values."""
    _require("cdsapi")
    import cdsapi

    url = url or os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
    key = key or os.environ.get("CDS_API_KEY", "")

    if not key:
        _print_api_key_instructions()
        raise SystemExit(1)

    return cdsapi.Client(url=url, key=key, quiet=False, progress=True)


def _print_api_key_instructions() -> None:
    print("""
  CDS API key not found.  ERA5 data requires a free Copernicus account.

  Steps to obtain your API key:
  1. Register at https://cds.climate.copernicus.eu  (free)
  2. Accept the ERA5 licence:
       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
       (click "Licences" tab -> accept)
  3. Go to your profile page to copy your API key:
       https://cds.climate.copernicus.eu/profile
  4. Add to .env in the repo root:
       CDS_API_KEY=your-api-key-here
       CDS_API_URL=https://cds.climate.copernicus.eu/api   # (optional, this is the default)

  Note: As of 2024 the CDS uses a new API endpoint (/api not /api/v2).
  If you have an old-format key (uid:hash), register again to obtain the new format.
""")


def _area_box(lat: float, lon: float, margin: float = 0.3) -> list[float]:
    """Return [N, W, S, E] bounding box around a point for ERA5 area request."""
    return [
        round(lat + margin, 4),
        round(lon - margin, 4),
        round(lat - margin, 4),
        round(lon + margin, 4),
    ]


def _all_months() -> list[str]:
    return [f"{m:02d}" for m in range(1, 13)]


def _all_days() -> list[str]:
    return [f"{d:02d}" for d in range(1, 32)]


def _all_hours() -> list[str]:
    return [f"{h:02d}:00" for h in range(24)]


# -- Derivation functions -----------------------------------------------------

def _radiation_to_wm2(accumulated_j_m2):
    """Convert ERA5 hourly accumulated radiation (J/m²) to mean W/m²."""
    import numpy as np
    return np.maximum(accumulated_j_m2 / 3600.0, 0.0)


def _dewpoint_to_rh(t2m_k, d2m_k):
    """Derive relative humidity (%) from 2m temperature and dewpoint (K).

    Uses the Magnus formula approximation (Buck 1981), accurate to ~0.1% RH
    for temperatures between -40 C and +60 C.
    """
    import numpy as np
    t  = t2m_k - 273.15   # Celsius
    td = d2m_k - 273.15
    rh = 100.0 * (
        np.exp((17.625 * td) / (243.04 + td)) /
        np.exp((17.625 * t)  / (243.04 + t))
    )
    return np.clip(rh, 0.0, 100.0)


def _compute_dni(fdir_j_m2, timestamps, lat: float, lon: float):
    """Derive DNI (W/m²) from ERA5 direct horizontal irradiance using pvlib SZA.

    ERA5 'total_sky_direct_solar_radiation_at_surface' (fdir) is the direct
    radiation on a horizontal surface = DNI * cos(SZA).
    Inverting: DNI = fdir / cos(SZA).  At night (SZA >= 90 deg), DNI = 0.

    Args:
        fdir_j_m2: Array of hourly accumulated direct horizontal radiation (J/m²).
        timestamps: pandas DatetimeIndex (UTC) for each hourly step.
        lat, lon: Location coordinates.

    Returns:
        Array of DNI values in W/m².
    """
    _require("pvlib")
    import numpy as np
    import pvlib

    fdir_wm2 = _radiation_to_wm2(fdir_j_m2)
    loc      = pvlib.location.Location(latitude=lat, longitude=lon)
    solpos   = loc.get_solarposition(timestamps)
    cos_sza  = np.cos(np.radians(solpos["apparent_zenith"].values))

    # Avoid divide-by-zero at night; use a small floor threshold
    dni = np.where(cos_sza > 0.01, fdir_wm2 / cos_sza, 0.0)
    return np.maximum(dni, 0.0)


# -- Core fetch function ------------------------------------------------------

def fetch_era5(
    lat: float,
    lon: float,
    year: int,
    label: str,
    force: bool = False,
) -> Optional[Path]:
    """Download one year of ERA5 hourly data for a single location.

    Args:
        lat, lon: Target coordinates.  ERA5 grid is 0.25 deg; nearest grid
            point within a 0.3 deg margin is used.
        year:  Calendar year to fetch.
        label: Filesystem-safe location label (used in output filename).
        force: Re-download even if the output CSV already exists.

    Returns:
        Path to the output CSV file, or None on failure.
    """
    _require("cdsapi")
    _require("xarray", install="xarray netCDF4")
    _require("pvlib")
    import numpy as np
    import pandas as pd
    import xarray as xr

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    nc_path  = DATA_RAW / f"era5_{label}_{year}.nc"
    csv_path = DATA_RAW / f"era5_{label}_{year}.csv"

    if csv_path.exists() and not force:
        print(f"  [CACHED]  {csv_path.name} -- skipping (--force to redownload)")
        return csv_path

    # -- Step 1: download from CDS -------------------------------------------
    print(f"\n  Fetching ERA5  label={label}  lat={lat}  lon={lon}  year={year}")
    print(f"  Variables: {len(ERA5_VARIABLES)} fields")
    print(f"  Output NC : {nc_path.name}")

    client = _cds_client()
    request = {
        "product_type": "reanalysis",
        "variable":     ERA5_VARIABLES,
        "year":         str(year),
        "month":        _all_months(),
        "day":          _all_days(),
        "time":         _all_hours(),
        "area":         _area_box(lat, lon),
        "format":       "netcdf",
    }

    t0 = time.time()
    client.retrieve(CDS_DATASET, request, str(nc_path))
    elapsed = time.time() - t0
    print(f"  Download complete ({elapsed:.0f}s)  ->  {nc_path.name}")

    # -- Step 2: load NetCDF and derive columns -------------------------------
    print("  Deriving model weather columns from ERA5 fields...")
    ds = xr.open_dataset(nc_path)

    # ERA5 may return a small area; take the nearest grid point to the target.
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    # Flatten time dimension to 1-D arrays
    times    = pd.DatetimeIndex(ds["valid_time"].values)
    ssrd     = ds["ssrd"].values.ravel()    # surface solar radiation downwards (J/m²)
    fdir     = ds["fdir"].values.ravel()    # direct horizontal radiation (J/m²)
    t2m      = ds["t2m"].values.ravel()     # 2m temperature (K)
    d2m      = ds["d2m"].values.ravel()     # 2m dewpoint temperature (K)
    u10      = ds["u10"].values.ravel()     # 10m U-wind (m/s)
    v10      = ds["v10"].values.ravel()     # 10m V-wind (m/s)
    ds.close()

    ghi = _radiation_to_wm2(ssrd)
    dhi = _radiation_to_wm2(ssrd - fdir)    # diffuse = total - direct horizontal
    dni = _compute_dni(fdir, times, lat, lon)

    temp_c     = t2m - 273.15
    wind_speed = np.sqrt(u10**2 + v10**2)
    rh         = _dewpoint_to_rh(t2m, d2m)

    df = pd.DataFrame({
        "datetime":        times,
        "GHI_W_m2":        ghi.round(2),
        "DNI_W_m2":        dni.round(2),
        "DHI_W_m2":        dhi.round(2),
        "Temp_C":          temp_c.round(3),
        "WindSpeed_m_s":   wind_speed.round(3),
        "RelHumidity_pct": rh.round(2),
        # Source metadata (informational)
        "lat":             lat,
        "lon":             lon,
        "era5_label":      label,
        "year":            year,
    })

    # ERA5 time is UTC; keep as-is (preprocess.py will handle tz alignment)
    df = df.sort_values("datetime").reset_index(drop=True)

    # -- Step 3: save CSV -----------------------------------------------------
    df.to_csv(csv_path, index=False)
    rows = len(df)
    print(f"  Saved CSV -> {csv_path.name}  ({rows:,} rows)")
    print(f"  GHI  : [{df['GHI_W_m2'].min():.1f}, {df['GHI_W_m2'].max():.1f}] W/m2")
    print(f"  DNI  : [{df['DNI_W_m2'].min():.1f}, {df['DNI_W_m2'].max():.1f}] W/m2")
    print(f"  DHI  : [{df['DHI_W_m2'].min():.1f}, {df['DHI_W_m2'].max():.1f}] W/m2")
    print(f"  Temp : [{df['Temp_C'].min():.1f}, {df['Temp_C'].max():.1f}] C")
    print(f"  Wind : [{df['WindSpeed_m_s'].min():.2f}, {df['WindSpeed_m_s'].max():.2f}] m/s")
    print(f"  RH   : [{df['RelHumidity_pct'].min():.1f}, {df['RelHumidity_pct'].max():.1f}] %")

    return csv_path


# -- PVDAQ system CSV loader --------------------------------------------------

def load_pvdaq_locations(csv_path: Path) -> list[tuple[str, float, float, list[int]]]:
    """Parse a PVDAQ system CSV into fetch targets.

    Expected columns: system_id, lat, lon, years_available (comma-separated).
    Returns list of (label, lat, lon, years) tuples.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    targets = []
    for _, row in df.iterrows():
        label = f"pvdaq_{int(row['system_id'])}"
        lat   = float(row["lat"])
        lon   = float(row["lon"])
        years = [int(y) for y in str(row["years_available"]).split(",")]
        targets.append((label, lat, lon, years))
    return targets


# -- Entry point --------------------------------------------------------------

def main(force: bool = False, systems_csv: Optional[Path] = None,
         location: Optional[str] = None, year: Optional[int] = None) -> None:
    print("=" * 60)
    print("  ERA5 FETCH")
    print("=" * 60)

    # Check API key early
    if not os.environ.get("CDS_API_KEY"):
        _print_api_key_instructions()
        sys.exit(1)

    # Build fetch targets
    if systems_csv:
        print(f"  Loading PVDAQ system locations from: {systems_csv}")
        targets = load_pvdaq_locations(systems_csv)
        print(f"  Found {len(targets)} systems")
    else:
        targets = DEFAULT_LOCATIONS
        print(f"  Using {len(targets)} default locations")

    # Filter to single location/year if specified
    if location:
        targets = [(lbl, la, lo, yrs) for lbl, la, lo, yrs in targets if lbl == location]
        if not targets:
            print(f"  [ERROR] Location '{location}' not found in target list")
            sys.exit(1)
    if year:
        targets = [(lbl, la, lo, [yr for yr in yrs if yr == year])
                   for lbl, la, lo, yrs in targets]
        targets = [(lbl, la, lo, yrs) for lbl, la, lo, yrs in targets if yrs]

    # Count total fetches
    total = sum(len(yrs) for _, _, _, yrs in targets)
    print(f"  Total fetch jobs : {total}")
    print(f"  Output directory : {DATA_RAW}")
    print(f"  Force redownload : {force}")
    print("=" * 60)

    results: list[tuple[str, int, bool]] = []
    for label, lat, lon, years in targets:
        for yr in years:
            try:
                path = fetch_era5(lat, lon, yr, label, force=force)
                results.append((label, yr, path is not None))
            except Exception as exc:
                print(f"  [ERROR] {label} {yr}: {exc}")
                results.append((label, yr, False))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    n_ok  = sum(ok for _, _, ok in results)
    n_err = len(results) - n_ok
    for label, yr, ok in results:
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] {label} {yr}")
    print(f"\n  {n_ok}/{len(results)} succeeded, {n_err} failed")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ERA5 hourly weather from Copernicus CDS.")
    parser.add_argument("--force",    action="store_true",
                        help="Re-download even if output CSV already exists.")
    parser.add_argument("--systems",  type=Path, default=None, metavar="CSV",
                        help="CSV of PVDAQ system locations (system_id, lat, lon, years_available).")
    parser.add_argument("--location", type=str, default=None,
                        help="Fetch only this location label (from default list or --systems CSV).")
    parser.add_argument("--year",     type=int, default=None,
                        help="Fetch only this year.")
    args = parser.parse_args()
    main(force=args.force, systems_csv=args.systems,
         location=args.location, year=args.year)
