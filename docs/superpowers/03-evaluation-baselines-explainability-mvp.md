# MVP Plan 3 — Evaluation, Baselines, and Explainability

## Purpose

This document defines the **global MVP scope** for judging whether the project’s
results are meaningful.

Its job is to answer:

> What counts as evidence that the transfer-learning baseline is useful, and
> what minimum evaluation framework must exist before claims are credible?

This is not yet the detailed execution plan.

---

## MVP objective

Create a **basic but credible evaluation framework** that can compare the model
against at least one baseline, report the core metrics, and provide interpretable
analysis at the home/city level.

---

## Current reality

What already exists:

- zero-shot NY evaluation path exists,
- MAE is reported in the current training flow,
- train/validation/test boundaries are now much clearer than before.

What is still weak:

- no persistence baseline is implemented,
- RMSE and skill score are not yet operational,
- per-home NY reporting does not yet exist,
- explainability is not yet defined,
- model claims in docs exceed the actual evidence currently produced.

---

## What must be addressed before writing the detailed implementation plan

### 1. Metric set for the MVP

Before detailed planning, lock the minimum metric set:

- MAE,
- RMSE,
- persistence-relative skill score,
- in-region vs out-of-region comparison.

### 2. Baseline scope for the MVP

Decide what is mandatory now versus later.

For the MVP, the likely minimum is:

- persistence baseline,
- optionally one simple neural baseline after persistence is stable.

### 3. Reporting grain

Lock how results are reported:

- overall source validation,
- overall NY zero-shot test,
- per-home NY summaries,
- region/city-level summaries where meaningful.

### 4. Explainability scope

This needs a clear MVP boundary.

Likely MVP explainability should focus on:

- modality ablations,
- feature sensitivity / importance,
- simple error analysis by home/city/weather conditions.

Heavy explainability work such as large SHAP pipelines should be treated as
conditional unless the baseline evaluation is already stable.

### 5. Evidence standard for transfer claims

Before detailed planning, define what can and cannot be claimed from the MVP.

Examples:

- can the project claim zero-shot transfer only?
- can it claim any California benefit despite the single-home limitation?
- what counts as improvement over baseline?

---

## Global work packages

### Work package A — Core evaluation metrics

- implement the minimum metric set,
- ensure metrics are reported consistently across validation and test,
- define output artifacts for result summaries.

### Work package B — Baseline framework

- implement persistence baseline,
- define how baselines use the same splits,
- optionally add one simple learned baseline if the team has time.

### Work package C — Result reporting

- create summary tables,
- produce per-home NY output,
- compare source-region validation vs NY zero-shot performance,
- flag uncertainty and limitations explicitly.

### Work package D — Explainability / interpretation

- define lightweight explainability for the MVP,
- run modality ablations,
- identify feature or condition-level importance,
- analyze where the model succeeds and fails.

### Work package E — Comparative analysis by city/home

- compare behavior across source vs target regions,
- summarize per-city or per-region patterns,
- identify homes or conditions with unusually large errors.

---

## Deliverables for this workstream

1. one evaluation protocol,
2. one persistence baseline,
3. one results table format,
4. one per-home NY analysis output,
5. one lightweight explainability package,
6. one limitations/claims guidance summary.

---

## Acceptance criteria

This workstream is ready when:

- every reported model result is compared to a named baseline,
- the minimum metrics are present,
- NY zero-shot performance is reported at both aggregate and home level,
- the project has a basic explanation of what seems to drive performance,
- the team can defend its claims with explicit evidence rather than qualitative
  impressions.

---

## What should not be over-scoped yet

Do **not** make this workstream responsible for:

- full interpretability research,
- polished publication-grade figures,
- large explainability benchmarking,
- multiple advanced baselines before persistence is stable,
- final paper-writing.

---

## Consequent next step

After this global plan is accepted, the next document should be a **granular PDD
for evaluation/baseline execution** under `docs/superpowers/` with specific
metrics, artifacts, scripts, and comparison tables to implement.
