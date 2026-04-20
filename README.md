# CS7180 Final Project — Multi-Modal Solar Generation Forecasting

**CS7180: Applied Deep Learning**
**Team:** Nathan Cunningham · Nikita Demidov · Chaitanya Agarwal

---

## Overview

We train a multi-modal Transformer model to forecast residential solar generation
(kWh per 15-minute interval, 1-hour horizon) from three input streams: real-time
weather observations, historical generation, and static site metadata.

The core research question is **geographic transfer**: train on Texas and California
sites, then evaluate zero-shot on New York sites — testing whether the model learns
weather-to-generation mappings that generalise across climates without fine-tuning.

---

## Model Architecture

![Model Architecture](docs/model_architecture.png)

### Component Descriptions

| Component | File | Description |
|-----------|------|-------------|
| **WeatherEncoder** | `src/models/encoders/weather.py` | Transformer encoder (2 layers, 4-head MHA, d=128) that processes a 96-step sequence of GHI, DNI, DHI, temperature, wind speed, and relative humidity. A learnable linear projection maps the 6 raw features to d=128, followed by sinusoidal positional encoding. |
| **GenerationEncoder** | `src/models/encoders/generation.py` | Identical Transformer architecture operating on the univariate past-kWh series. Shares the same d=128 / 4-head / 2-layer configuration, keeping the two latent spaces compatible for cross-attention. |
| **MetadataEncoder** | `src/models/encoders/metadata.py` | Two-layer MLP (6→64→128) with LayerNorm and ReLU that embeds time-invariant site features — latitude, longitude, panel tilt, azimuth, system capacity, and elevation — into the same d=128 space. |
| **CrossAttentionFusion** | `src/models/fusion/cross_attention.py` | `nn.MultiheadAttention` (4 heads) with the **generation sequence as Query** and the **weather sequence as Key/Value**. A residual connection and LayerNorm are applied before mean-pooling over the sequence dimension to produce a single (B, 128) vector. |
| **Concatenate** | `src/models/__init__.py` | The pooled fusion vector (B, 128) is concatenated with the metadata embedding (B, 128), yielding a (B, 256) representation that combines dynamic weather-generation context with static site properties. |
| **RegressionHead** | `src/models/heads/regression.py` | Two-layer MLP (256→128→4) with ReLU and dropout that maps the fused representation to 4 predicted kWh values, one per 15-minute interval in the forecast horizon (1 hour total). |

### Input Data Streams

| Stream | Shape | Source | Features |
|--------|-------|--------|----------|
| **Weather time-series** | (B, 96, 6) | NSRDB GOES CONUS v4 | GHI, DNI, DHI, air temperature, wind speed, relative humidity |
| **Generation history** | (B, 96, 1) | Pecan Street Dataport | Past solar generation (kWh, 15-min intervals) |
| **Site metadata** | (B, 6) | Pecan Street + PVDAQ | Latitude, longitude, panel tilt, azimuth, system capacity (kW), elevation (m) |

Context window: **96 steps = 24 hours** at 15-minute resolution.
Forecast horizon: **4 steps = 1 hour** ahead.

### Key Hyperparameters (`configs/experiment.yaml`)

| Parameter | Value |
|-----------|-------|
| `d_model` | 128 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `ffn_dim` | 256 |
| `dropout` | 0.1 |
| `seq_len` | 96 |
| `forecast_horizon` | 4 |
| Total parameters | ~640K |

---

## Transfer Learning Evaluation Strategy

The model is trained exclusively on **Texas (Austin)** and **California (San Jose)**
residential PV sites from the Pecan Street dataset. New York sites are held out
entirely during training and used only for final evaluation.

**Rationale:** Austin and California represent sunny, dry climates with high and
consistent irradiance. New York has a cloudier, more variable continental climate.
If the model generalises — without any New York examples — it suggests the
cross-attention mechanism learns transferable weather-to-generation physics rather
than site-specific patterns.

**Evaluation metrics:** MAE (kWh), RMSE (kWh), and skill score relative to a
persistence baseline (predict next hour = current hour).

| Split | Sites | Region |
|-------|-------|--------|
| Train (70%) | Austin + California homes | Texas, California |
| Validation (15%) | Austin + California homes | Texas, California |
| **Test (zero-shot)** | **New York homes** | **New York** |

---

## Repository Structure

```
CS7180_final/
├── configs/
│   └── experiment.yaml        # All hyperparameters and Trainer config
├── data/raw/                  # Large CSVs (Git LFS tracked)
│   ├── pecanstreet_*          # Generation data (1-min, 15-min)
│   ├── nsrdb_*                # NREL NSRDB weather data
│   └── nasa_power_*           # NASA POWER backup weather data
├── docs/
│   └── model_architecture.png # Architecture diagram
├── src/
│   ├── dataloader.py          # Polars-backed SolarWindowDataset + LightningDataModule
│   ├── fetch_pecanstreet.py   # Pecan Street Dataport fetch script
│   ├── fetch_nsrdb.py         # NREL NSRDB fetch script
│   ├── fetch_nasa_power.py    # NASA POWER fetch script
│   ├── fetch_pvdaq.py         # DOE PVDAQ S3 metadata + candidate selection
│   └── models/
│       ├── __init__.py        # SolarForecastModel (LightningModule) + build()
│       ├── base.py            # Abstract base classes
│       ├── encoders/
│       │   ├── weather.py     # Transformer encoder for weather
│       │   ├── generation.py  # Transformer encoder for generation history
│       │   └── metadata.py    # MLP encoder for static site features
│       ├── fusion/
│       │   └── cross_attention.py  # Cross-attention + mean pool
│       └── heads/
│           └── regression.py  # MLP regression head
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Chaim3ra/CS7180_final
cd CS7180_final
conda create -n CS7180 python=3.11
conda activate CS7180
pip install -r requirements.txt
pip install dvc-s3
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# edit .env — required fields:
#   NREL_API_KEY       — from developer.nrel.gov
#   AWS_ACCESS_KEY_ID  — IAM key with s3:GetObject on cs7180-final-project
#   AWS_SECRET_ACCESS_KEY
```

Configure DVC S3 credentials locally (never committed):

```bash
dvc remote modify --local s3remote access_key_id     <AWS_ACCESS_KEY_ID>
dvc remote modify --local s3remote secret_access_key <AWS_SECRET_ACCESS_KEY>
```

Pull all data (~3 GB) from S3:

```bash
dvc pull
```

---

## Quick Start

```python
from src.models import build

model = build("configs/experiment.yaml")
print(model)  # full architecture summary
```
