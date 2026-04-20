# Data Documentation

## Sources

| File | Region | Source | Coverage |
|------|--------|--------|----------|
| `pecanstreet_austin_15min_solar.csv` | Texas | Pecan Street Dataport | 19 homes, 2018 |
| `pecanstreet_california_15min_solar.csv` | California | Pecan Street Dataport | 4 homes exported; 1 usable (dataid 9836), 2014-2015 |
| `pecanstreet_newyork_15min_solar.csv` | New York | Pecan Street Dataport | 14 homes, 2019 |
| `nasa_power_austin_2018.csv` | Texas | NASA POWER | Hourly, 2018 |
| `nasa_power_california_2014.csv` | California | NASA POWER | Hourly, 2014 |
| `nasa_power_california_2015.csv` | California | NASA POWER | Hourly, 2015 |
| `nasa_power_newyork_2019.csv` | New York | NASA POWER | Hourly, 2019 |
| `pecanstreet_metadata.csv` | All | Pecan Street Dataport | Home metadata (city, tilt, azimuth, capacity) |

Weather features: GHI, DNI, DHI (W/m2), air temperature (C), wind speed (m/s), relative humidity (%).

## CA Data Quality Issue

The California dataset has a significant data gap:

- **135 CA homes** appear in `pecanstreet_metadata.csv`
- **62 homes** are flagged `pv=yes` in metadata (i.e., have solar panels)
- **4 homes** were actually exported in `pecanstreet_california_15min_solar.csv`
- **3 of those 4** (dataids 1731, 4495, 8342) have `solar=0.0` for every row in 2018 — broken meter readings, filtered by the 5% non-zero threshold
- **1 home** (dataid 9836) has valid solar data: 2014-2015, 75.6% non-zero

The 62 homes with `pv=yes` that are absent from the export were never included in the Dataport data pull. This is a data collection gap, not a processing bug.

## Training Set Imbalance

| Region | Homes | Rows | Split |
|--------|-------|------|-------|
| Texas (Austin) | 19 | ~660K | Train |
| California (San Jose) | 1 | ~34K | Train |
| New York | 14 | ~500K | Test (zero-shot) |

The training set is **TX-dominant** (~95% of training rows from Texas). The single CA home provides some climate diversity (San Jose vs Austin) but the model will be biased toward Texas irradiance patterns.

## Recommended Fix for Next Iteration

To address the CA data gap, augment with **DOE PVDAQ** California systems. The `src/fetch_pvdaq.py` script already identifies PVDAQ candidate systems filtered by state. PVDAQ provides 15-minute PV production data from real residential and commercial installations across the US, including many CA sites.

Steps:
1. Run `python src/fetch_pvdaq.py` to download CA candidate system list
2. Select systems with multi-year coverage and >5% non-zero generation
3. Fetch NASA POWER weather for matching CA lat/lon coordinates
4. Add to `src/preprocess.py` as a `ca_pvdaq` region alongside the Pecan Street CA home

This would provide geographic and climate diversity (multiple CA cities: San Diego, Los Angeles, Sacramento) that the current single-home CA dataset lacks.
