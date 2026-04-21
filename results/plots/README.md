# results/plots

Each `.png` has a companion `.csv` with the exact data used to generate it.
Re-create any plot without re-running the model:

| PNG | CSV | Description |
|-----|-----|-------------|
| data_efficiency_curve.png | data_efficiency_curve.csv | MAE vs NY fine-tuning days (0/7/30/90) |
| metric_comparison.png | metric_comparison.csv | All 7 metrics, all experiments, side-by-side bars |
| generalization_gap.png | generalization_gap.csv | In-region vs out-region MAE per experiment |
| skill_score_curve.png | skill_score_curve.csv | Skill score vs persistence, by fine-tuning days |

**Color scheme:** M1 (real data only) = blue (#1f77b4) · M2 (real + synthetic) = orange (#ff7f0e)
**Line style:** Zero-shot = solid · Fine-tuned = dashed
