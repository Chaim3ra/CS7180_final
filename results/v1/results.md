# V1 Training Results

## Run Details

| Field | Value |
|-------|-------|
| Run ID | 2026-04-21_01-40-26 |
| Hardware | NVIDIA A100-SXM4-40GB |
| Precision | fp16 mixed precision |
| Training time | ~13 minutes |
| Best checkpoint | `s3://cs7180-final-project/checkpoints/2026-04-21_01-40-26/solar-epoch=04-val_loss=0.0042.ckpt` |

## Model

| Parameter | Value |
|-----------|-------|
| Total parameters | 639,940 |
| d_model | 128 |
| Encoder layers | 2 |
| Attention heads | 4 |
| Forecast horizon | 4 steps (1 hour) |
| Context window | 96 steps (24 hours) |
| Batch size | 256 |

## Dataset

| Split | Homes | Region | Windows |
|-------|-------|--------|---------|
| Train (85%) | 19 TX (2018) + 1 CA (2014–2015) | Austin + San Jose | 588,185 |
| Val (15%) | 19 TX + 1 CA | Austin + San Jose | 102,176 |
| Test (zero-shot) | 14 NY (2019) | New York | 245,909 |

Weather source: NASA POWER reanalysis (~50 km resolution).

## Results

| Metric | Value |
|--------|-------|
| In-region MAE (TX+CA val) | **0.0295 kWh** |
| Out-of-region MAE (NY zero-shot) | **0.1657 kWh** |
| Generalization gap | **+0.1363 kWh (5.6× worse on NY)** |

## Training Curve

Early stopping triggered at epoch 15. Best checkpoint at epoch 5 (val_loss: 0.0042).

| Epoch | train_loss | val_loss | MAE |
|-------|-----------|----------|-----|
| 1 | 0.0071 | 0.0043 | 0.0312 |
| 2 | 0.0063 | 0.0043 | 0.0300 |
| 3 | 0.0048 | 0.0043 | 0.0294 |
| 4 | 0.0110 | 0.0044 | 0.0305 |
| **5** | **0.0036** | **0.0042** | **0.0295** |
| 6 | 0.0046 | 0.0044 | 0.0296 |
| 7 | 0.0049 | 0.0048 | 0.0302 |
| 8 | 0.0047 | 0.0048 | 0.0313 |
| 9 | 0.0036 | 0.0051 | 0.0312 |
| 10 | 0.0047 | 0.0046 | 0.0296 |
| 11 | 0.0046 | 0.0052 | 0.0309 |
| 12 | 0.0041 | 0.0057 | 0.0319 |
| 13 | 0.0038 | 0.0054 | 0.0308 |
| 14 | 0.0049 | 0.0058 | 0.0331 |
| 15 | 0.0037 | 0.0056 | 0.0321 |

## Known Limitations

- **Training data too small:** Only 20 homes across 2 regions. Limited geographic and climate diversity.
- **CA severely underrepresented:** 1 CA home (dataid 9836) due to Pecan Street data quality issues — 62 CA homes absent from Dataport export, 3 others filtered for all-zero solar.
- **Single year per region:** 1 year of data per region misses inter-annual weather variability (El Niño, drought years, etc.).
- **Low-resolution weather:** NASA POWER reanalysis at ~50 km spatial resolution introduces 10–15% RMSE vs. ground-truth irradiance measurements.
- **Large generalization gap (5.6×):** NY climate (cloudy, variable, continental) is substantially different from TX/CA (sunny, arid). The model has not seen this regime.

## Motivation for V2

The 5.6× generalization gap motivates the V2 expansion:

- **PVDAQ data:** 50+ systems across CA, TX, NM, AZ, CO with 5–10 years of data — increases training diversity by ~2 orders of magnitude
- **ERA5 weather:** Copernicus CDS reanalysis at 0.25° resolution with additional features (cloud cover, pressure, precipitable water) — more accurate than NASA POWER
- **Larger model:** d_model 128→256, seasonal positional encoding, learned site embeddings
