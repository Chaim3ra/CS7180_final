# Solar Forecasting — All Experiment Results

*Generated: 2026-04-21 02:41:31*

## Metric Definitions

- **mae**: Mean absolute error (kWh) — average magnitude of prediction error
- **rmse**: Root mean squared error (kWh) — penalises large errors more than MAE
- **mape**: Mean absolute percentage error (%) — relative error; zero-actual steps excluded
- **r2**: Coefficient of determination — fraction of variance explained (1.0 = perfect)
- **skill_score**: Skill score vs persistence (%) — positive means better than same-time-yesterday
- **peak_mae**: Peak-hour MAE (kWh) — MAE restricted to 8 am–4 pm solar production hours
- **generalization_gap**: Generalisation gap (kWh) — out-of-region MAE minus in-region MAE

## Summary Table (sorted by MAE)

| model_version | experiment | ny_days | mae | rmse | r2 | skill_score | peak_mae |
|---|---|---|---|---|---|---|---|
| v1 | in_region_ca | 0 | 0.0173 | 0.0333 | 0.946 | 8.618 | 0.0344 |
| v1 | in_region_tx | 0 | 0.0356 | 0.0739 | 0.939 | 55.669 | 0.0779 |
| v1 | zero_shot | 0 | 0.1657 | 0.3765 | 0.516 | 28.993 | 0.3338 |

## v1

### in_region_ca (ny_days=0)

- mae: 0.0173
- rmse: 0.0333
- mape: 352.7248
- r2: 0.9459
- skill_score: 8.6180
- peak_mae: 0.0344
- generalization_gap: 
- timestamp: 2026-04-21T02:19:13
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/2026-04-21_01-40-26/solar-epoch=04-val_loss=0.0042.ckpt

### in_region_tx (ny_days=0)

- mae: 0.0356
- rmse: 0.0739
- mape: 77.6902
- r2: 0.9394
- skill_score: 55.6688
- peak_mae: 0.0779
- generalization_gap: 
- timestamp: 2026-04-21T02:41:31
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/2026-04-21_01-40-26/solar-epoch=04-val_loss=0.0042.ckpt

### zero_shot (ny_days=0)

- mae: 0.1657
- rmse: 0.3765
- mape: 606.1434
- r2: 0.5157
- skill_score: 28.9935
- peak_mae: 0.3338
- generalization_gap: 
- timestamp: 2026-04-21T02:29:28
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/2026-04-21_01-40-26/solar-epoch=04-val_loss=0.0042.ckpt

