# Limitations and Future Work

## Infrastructure and Pipeline

**Preprocessing is not decoupled from training.** The current pipeline re-runs preprocessing on every training invocation, which is wasteful and fragile. Preprocessing should be treated as a one-time offline step whose outputs are cached in S3 as versioned processed parquets; training scripts should read directly from those parquets without triggering any raw-data processing. This decoupling would also make it easier to reproduce results on new hardware without re-downloading raw data.

**The Colab training notebook has not been updated for the V2 branch.** The notebook targets `main` and has no awareness of `dev/v2-synthetic-ca` or its config differences. A proper solution would either introduce a single branch-aware notebook that accepts a `MODEL_VERSION` cell parameter and selects the correct config, checkpoint prefix, and results path at runtime, or maintain a separate `train_colab_v2.ipynb` with V2-specific defaults.

**There is no resume-from-checkpoint support.** If a training run crashes or is preempted mid-epoch, the entire run must be restarted from epoch 0. Adding Lightning's `resume_from_checkpoint` path to the trainer configuration would make long runs on preemptible cloud instances viable and reduce wasted compute.

**A git rebase-merge directory conflict surfaces in Colab when cloning the repository.** This points to a fragile automated git workflow that is not suitable for distributed or headless training environments. A cleaner approach would pin training runs to a specific commit SHA or release tag rather than a branch tip, eliminating merge state entirely from the training environment.

---

## Evaluation and Results

**M2 zero-shot metrics were not automatically persisted by `train.py`.** Zero-shot performance was computed during the V2 training run but not written to `all_results.csv`; it had to be re-evaluated separately using `evaluate.py`. The training script should write zero-shot metrics to the results CSV before beginning fine-tuning, so that the evaluation record is complete and reproducible without a second inference pass.

**MAPE values are substantially inflated by near-zero nighttime solar readings.** Reported MAPE values range from roughly 600% to 1,300% across experiments, making this metric nearly uninterpretable. The root cause is that mean absolute percentage error is ill-conditioned when the true value approaches zero, which occurs at every nighttime timestep. Future evaluations should either exclude timesteps where ground-truth generation falls below a minimum threshold (e.g., 0.01 kWh per 15-min interval) or replace MAPE with a scale-independent metric such as normalized MAE relative to installed capacity.

**All reported metrics are single-seed point estimates with no measure of variance.** Every experiment was run once with a fixed random seed, so it is not possible to distinguish meaningful differences from run-to-run variation. Future work should report mean and standard deviation across at least three independent seeds for each configuration, and apply appropriate significance tests when comparing V1 and V2 models.

---

## Data

**Historical California weather relies on NASA POWER reanalysis rather than a satellite-derived product.** NSRDB GOES CONUS, the preferred source for high-resolution surface irradiance, covers 2018 onward and therefore cannot provide weather data for the 2014–2015 Pecan Street CA measurement period. NASA POWER reanalysis was used as a fallback at ~50 km spatial resolution. ERA5 reanalysis at 31 km resolution, available from 1940 onward, would provide higher temporal fidelity and is routinely used in solar resource assessment; it represents a straightforward improvement over the current weather inputs.

**Only one real California home is available from Pecan Street.** Home 9836 (San Diego, 2014–2015) is the sole CA PV home whose generation data was included in the Dataport export. The 18 synthetic homes generated in this work partially compensate for this scarcity but cannot replicate the full diversity of real measured data. Pecan Street metadata lists 62 additional CA homes with solar panels; obtaining the missing Dataport export for these homes would substantially increase the real CA training set and reduce reliance on synthetic augmentation.

**PVDAQ integration was planned but not completed.** The NREL PVDAQ database contains measured generation data from more than 50 geographically diverse residential and commercial PV systems across Florida, Colorado, Arizona, Nevada, and New Mexico. Integrating PVDAQ would broaden the training distribution to cover multiple climate zones and reduce the geographic concentration on Texas and California that characterizes the current dataset.

**Synthetic panel parameters are drawn from population-level distributions rather than home-specific records.** System size, tilt, and azimuth for the 18 synthetic San Diego homes were sampled from the Tracking the Sun 2024 San Diego RES_SF empirical distribution. While this reproduces the statistical properties of installed systems, it does not reflect the actual specifications of the Civita development homes used as hosts. Access to permit records or utility interconnection data for the specific homes would allow more accurate per-home parameterization.

**Each region is represented by a single calendar year of data.** Texas covers 2018, California covers 2014–2015, and New York covers 2019. Single-year snapshots miss inter-annual variability in solar resource (driven by El Niño/La Niña cycles, volcanic aerosols, and decadal climate trends) and do not capture panel degradation over time, which averages roughly 0.5% per year for crystalline silicon modules. Multi-year data would yield more robust estimates of model generalization across weather regimes.

---

## Model

**The effect of synthetic home count on transfer performance has not been ablated.** The choice of 18 synthetic homes was motivated by achieving a roughly balanced TX/CA training mix, but the sensitivity of zero-shot and fine-tuned NY performance to this count (e.g., 10, 18, 36, or all 62 potential CA homes) was never tested. A systematic ablation would clarify whether synthetic augmentation offers diminishing returns beyond a threshold and whether the observed V2 zero-shot regression relative to V1 is attributable to count, geographic mismatch, or synthetic noise characteristics.

**No hyperparameter search was performed.** Model capacity parameters (`d_model`, `nhead`, `num_layers`) and the learning rate were inherited from a manually chosen baseline and held fixed across both V1 and V2 experiments. Bayesian optimization or a structured grid search over these parameters, cross-validated on the NY zero-shot set, could yield meaningful improvements without architectural changes.

**Fine-tuning freezes the weather and generation encoders.** The current fine-tuning strategy updates only the forecast head while keeping all encoder weights frozen. Alternative strategies — full fine-tuning with a lower learning rate, lightweight adapter layers inserted into the encoder, or domain adversarial training that encourages encoder representations to be region-invariant — have been shown to improve transfer in related time-series forecasting settings and warrant systematic comparison.

**Zero-shot transfer has only been evaluated on New York.** The model was trained on Texas and California data and evaluated on New York as a held-out region, but it is unknown whether the observed generalization pattern holds for other unseen climates. Evaluating on Florida, Colorado, or Arizona — regions with distinct irradiance regimes, temperature profiles, and panel tilt conventions — would provide a more complete picture of the model's geographic generalization envelope.

**Synthetic CA data degrades zero-shot transfer to New York.** M2 zero-shot MAE (0.171 kWh) is slightly worse than M1 (0.166 kWh), and M2 skill score (26.6%) lags M1 (29.0%) despite additional training data. The root cause has not been fully investigated. One plausible explanation is climate-regime conflict: San Diego has an arid Mediterranean climate with high, stable irradiance that is systematically different from New York's humid continental climate, and adding San Diego-like patterns to the training distribution may shift encoder representations in a direction that reduces NY zero-shot accuracy. Disentangling this effect from confounds such as noise calibration artifacts or schema differences between real and synthetic rows is an important direction for future work.
