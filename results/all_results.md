# Solar Forecasting — All Experiment Results

*Generated: 2026-04-21 17:47:16*

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
| v1 | finetune_180d | 180 | 0.0550 | 0.1524 | 0.721 | 50.826 |  |
| v1 | finetune_90d | 90 | 0.1046 | 0.2157 | 0.811 | 41.961 |  |
| v2 | finetune_90d | 90 | 0.1089 | 0.2200 | 0.804 | 39.591 |  |
| v1 | finetune_30d | 30 | 0.1325 | 0.2599 | 0.758 | 37.268 |  |
| v2 | finetune_30d | 30 | 0.1489 | 0.2820 | 0.715 | 29.482 |  |
| v1 | finetune_7d | 7 | 0.1495 | 0.2950 | 0.697 | 34.820 |  |
| v2 | finetune_7d | 7 | 0.1609 | 0.3124 | 0.660 | 29.864 |  |
| v1 | zero_shot | 0 | 0.1657 | 0.3765 | 0.516 | 28.994 | 0.3338 |
| v2 | zero_shot | 0 | 0.1714 | 0.3935 | 0.471 | 26.554 | 0.3682 |

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

### finetune_180d (ny_days=180)

- mae: 0.0550
- rmse: 0.1524
- mape: 953.8498
- r2: 0.7210
- skill_score: 50.8259
- peak_mae: 
- generalization_gap: 
- epoch_stopped: 13
- timestamp: 2026-04-21T08:39:36
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/finetune_v1_ny180days/ft-v1-ny180d-epoch=06-val_loss=0.0232.ckpt

### finetune_90d (ny_days=90)

- mae: 0.1046
- rmse: 0.2157
- mape: 826.5948
- r2: 0.8115
- skill_score: 41.9615
- peak_mae: 
- generalization_gap: 
- epoch_stopped: 14
- timestamp: 2026-04-21T08:15:10
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/finetune_v1_ny90days/ft-v1-ny90d-epoch=07-val_loss=0.0465.ckpt

### finetune_30d (ny_days=30)

- mae: 0.1325
- rmse: 0.2599
- mape: 1021.8704
- r2: 0.7579
- skill_score: 37.2680
- peak_mae: 
- generalization_gap: 
- epoch_stopped: 11
- timestamp: 2026-04-21T08:11:19
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/finetune_v1_ny30days/ft-v1-ny30d-epoch=04-val_loss=0.0675.ckpt

### finetune_7d (ny_days=7)

- mae: 0.1495
- rmse: 0.2950
- mape: 1326.8901
- r2: 0.6969
- skill_score: 34.8201
- peak_mae: 
- generalization_gap: 
- epoch_stopped: 13
- timestamp: 2026-04-21T08:08:14
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/finetune_v1_ny7days/ft-v1-ny7d-epoch=06-val_loss=0.0870.ckpt

### zero_shot (ny_days=0)

- mae: 0.1657
- rmse: 0.3765
- mape: 606.1435
- r2: 0.5157
- skill_score: 28.9935
- peak_mae: 0.3338
- generalization_gap: 
- timestamp: 2026-04-21T08:39:36
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/2026-04-21_01-40-26/solar-epoch=04-val_loss=0.0042.ckpt

## v2

### finetune_90d (ny_days=90)

- mae: 0.1089
- rmse: 0.2200
- mape: 909.0188
- r2: 0.8038
- skill_score: 39.5910
- peak_mae: 
- generalization_gap: 
- epoch_stopped: 11
- timestamp: 2026-04-21T17:47:16
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/finetune_v2_ny90days/ft-v2-ny90d-epoch=04-val_loss=0.0484.ckpt

### finetune_30d (ny_days=30)

- mae: 0.1489
- rmse: 0.2820
- mape: 1727.6283
- r2: 0.7149
- skill_score: 29.4819
- peak_mae: 
- generalization_gap: 
- epoch_stopped: 11
- timestamp: 2026-04-21T17:43:58
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/finetune_v2_ny30days/ft-v2-ny30d-epoch=04-val_loss=0.0795.ckpt

### finetune_7d (ny_days=7)

- mae: 0.1609
- rmse: 0.3124
- mape: 1278.0969
- r2: 0.6601
- skill_score: 29.8637
- peak_mae: 
- generalization_gap: 
- epoch_stopped: 16
- timestamp: 2026-04-21T17:34:01
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/finetune_v2_ny7days/ft-v2-ny7d-epoch=09-val_loss=0.0976.ckpt

### zero_shot (ny_days=0)

- mae: 0.1714
- rmse: 0.3935
- mape: 663.0951
- r2: 0.4710
- skill_score: 26.5535
- peak_mae: 0.3682
- generalization_gap: 
- timestamp: 2026-04-21T17:47:16
- checkpoint_s3_path: s3://cs7180-final-project/checkpoints/2026-04-21_09-21-34/solar-epoch=02-val_loss=0.0035.ckpt

