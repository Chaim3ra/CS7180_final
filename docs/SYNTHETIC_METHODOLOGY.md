# Synthetic CA Solar Generation — Methodology

**Module:** `src/synthetic.py`
**Config:** `configs/experiment_v2.yaml` (under `synthetic:`)

---

## Why synthetic data?

The Pecan Street Dataport contains only one San Diego home with measured solar generation (home 9836). One home cannot represent the geographic and system-configuration diversity needed to train a generalisable model. Rather than excluding California from the training set entirely, we generate 18 synthetic San Diego homes whose solar profiles are grounded in three empirical sources:

1. Real building stock from the Pecan Street Civita cohort (57 homes)
2. Real installed-system parameters from 274,000 San Diego residential installations (LBNL Tracking the Sun 2024)
3. A physics-validated simulation model (pvlib PVWatts) driven by NASA POWER reanalysis weather

The result is a CA training split with realistic climate, system-size, and orientation diversity that a single measured home cannot provide.

---

## Step 1 — Host home selection

We draw 18 homes from the 57 Pecan Street San Diego (Civita development) homes, stratified by building type in proportion to the full Civita distribution:

| Building type | Full Civita | Synthetic sample |
|---|---|---|
| Apartment | 28 (49%) | 9 |
| Town Home | 23 (40%) | 7 |
| Single-Family Home | 6 (11%) | 2 |

Each host home contributes only its building type. All other attributes (dataid, square footage, construction year) are not used — they are either unavailable or irrelevant to solar generation modelling.

**Why stratify?** Roof area and roof geometry differ systematically by building type. Sampling by type ensures the synthetic population reflects the real mix of housing in Civita rather than accidentally over-representing apartments (the largest group).

---

## Step 2 — Panel parameter sampling

For each synthetic home, we independently sample system size, tilt, and azimuth from the empirical distribution of 274,406 San Diego county residential single-family (RES_SF) installations from the LBNL Tracking the Sun 2024 dataset.

We restrict to the **p10–p90 range** of each parameter to exclude extreme outliers (very large commercial systems, data entry errors) while preserving real-world variability.

### System size

System size is additionally stratified by building type, reflecting the physical constraint that roof area limits panel count:

| Building type | Size sampling range |
|---|---|
| Apartment | p10–p50 (3.2–5.4 kW typical) |
| Town Home | p10–p90 (full range, 3.2–8.0 kW) |
| Single-Family Home | p50–p90 (5.4–8.0 kW typical) |

### Tilt

Tilt is drawn from p10–p90 (approximately 5°–25°) for all building types. San Diego rooftops tend to be shallower than the national average because the region's high irradiance makes south-facing tilt less critical, and flat/low-slope roofs are common in modern San Diego construction (median tilt = 18°).

### Azimuth

Azimuth is drawn from p10–p90 (approximately 90°–250°, i.e. east through south to slightly west). Many San Diego homes face east or west due to street grid orientation in planned developments like Civita.

### Module type

Module technology is sampled from the empirical Tracking the Sun distribution: 75% monocrystalline silicon, 18% multicrystalline silicon, weighted to give 80.7% mono-c-Si probability (i.e. mono / (mono + multi)).

### Location

Each home is assigned a unique location within the Civita development:

```
lat = 32.7849 ± Uniform(0, 0.005°)    # ≈ ±550 m
lon = -117.1539 ± Uniform(0, 0.005°)
```

The jitter keeps homes distinct for pvlib solar position calculations while keeping all homes within the physical bounds of the Civita campus.

---

## Step 3 — Physics-based generation (pvlib PVWatts)

We use the **pvlib PVWatts model** to convert weather inputs into AC power output. PVWatts is the U.S. DOE standard model for residential PV estimation, developed and validated by NREL against measured generation data across U.S. climate zones.

### Weather inputs

NASA POWER provides hourly surface meteorology at ~50 km spatial resolution. The three irradiance components (GHI, DNI, DHI) and air temperature are linearly interpolated from hourly to 15-minute resolution to match the Pecan Street 15-min output. Irradiance is clipped to ≥ 0 W/m² after interpolation to prevent physically impossible negative values.

**Date range:** 2014-07-08 through 2015-06-30, exactly matching the date coverage of real home 9836.

### PVWatts DC model

```
P_DC = P_DC0 × (GHI_POA / 1000) × [1 + γ × (T_cell − 25)]
```

Where:
- `P_DC0` = nameplate DC capacity (system_size_kw × 1000 W)
- `GHI_POA` = plane-of-array irradiance from the Hay-Davies transposition model (W/m²)
- `γ = −0.004 %/°C` = temperature coefficient for crystalline silicon
- `T_cell` = cell temperature from the SAPM thermal model

### PVWatts AC model

```
P_AC = min(P_DC × η_inv, P_DC0)
```

Where `η_inv = 0.96` (nominal inverter efficiency for a modern string inverter).

### Temperature model

We use the SAPM open-rack glass-polymer temperature model with parameters `a = −3.56, b = −0.075, ΔT = 3`. Open-rack (rather than close-roof-mount) is a conservative choice that predicts slightly higher cell temperatures. For San Diego's relatively moderate summer temperatures the bias is ≤ 2% on annual yield, which is within the uncertainty of the 50 km resolution weather data.

---

## Step 4 — Noise calibration

The pvlib PVWatts output is deterministic and smooth. Real solar generation contains variance from sub-hourly cloud events, soiling, and inverter transients. To make synthetic data realistic, we add calibrated Gaussian noise.

**Procedure:**
1. Load real home 9836 (the only measured San Diego PV home) at 1-min resolution.
2. Resample to 15-min mean.
3. Compute the standard deviation of generation for each clock hour (0–23).
4. For each synthetic 15-min step, draw noise from `N(0, σ_h)` where `σ_h` is the real home's std for hour `h`.
5. Clip the result to ≥ 0 (negative generation is physically impossible).

This approach preserves the time-of-day structure of real variance: midday cloud-passing events produce the largest noise, while early morning and evening hours (low irradiance) produce small noise. The `noise_scale` config parameter (default 1.0) can compress or amplify the noise magnitude.

---

## Limitations

1. **Single weather location.** All 18 homes share the same NASA POWER grid cell (~50 km resolution). Within-city microclimate variation (coastal fog vs. inland heat) is not captured.

2. **No shading model.** Urban tree and building shading is not modelled. Civita apartments likely experience more shading than simulated, leading to a slight overestimate of apartment generation.

3. **Fixed inverter efficiency.** The PVWatts model uses a constant `η_inv = 0.96`. Real inverters have efficiency curves that drop at low load; morning and evening generation may be slightly overestimated.

4. **Noise from a single reference home.** The calibration noise profile comes from home 9836 alone. A single home's variance may not generalise — cloudier days or specific system faults could distort the profile. With more real CA homes this would average out.

5. **No battery or export clipping.** Homes with batteries or small grid connections that clip export peaks are not modelled. Peak generation may be slightly overestimated for large systems.

---

## Reproducibility

All sampled parameters are logged to `data/processed/synthetic_ca_parameters.csv` with the generation timestamp, random seed, pvlib version, and numpy version in the file header. The NumPy Generator state is saved alongside this file. Given the same seed and library versions, running

```bash
python src/synthetic.py --config configs/experiment_v2.yaml --seed 42
```

will produce byte-identical output.
