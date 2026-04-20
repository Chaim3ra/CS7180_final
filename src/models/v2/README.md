# V2 Architecture Notes

This directory is for architecture changes introduced in the v2 model (branch `dev/v2-pvdaq`).

## Planned changes

The v1 architecture (`src/models/`) remains unchanged. V2 changes will be developed here and
integrated into the main model package once validated.

### Candidate improvements

- **Larger encoder capacity** — increase `d_model` from 128 to 256, or add a third Transformer layer, to handle the larger and more diverse PVDAQ training set
- **Seasonal positional encoding** — replace sinusoidal positional encoding with a learnable encoding that includes day-of-year and hour-of-day as additional inputs, giving the model explicit access to solar geometry
- **ERA5 feature expansion** — ERA5 provides additional variables (cloud cover fraction, surface pressure, precipitable water) not available from NASA POWER; evaluate whether adding these improves accuracy
- **Site embedding** — replace the fixed metadata MLP with a learned site embedding that is updated during training, allowing the model to capture installation-specific behavior beyond lat/lon/tilt/azimuth

## File layout (to be populated)

```
src/models/v2/
├── __init__.py          # exports updated SolarForecastModelV2 and build_v2()
├── README.md            # this file
├── encoders/            # any modified encoder variants
└── heads/               # any modified head variants
```

## Status

Not yet implemented. V1 full training must complete first to establish the baseline
MAE and generalization gap before V2 changes are worth building.
