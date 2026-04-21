# MVP Plan 2 — Model Architecture, Data Loading, and Training

## Purpose

This document defines the **global MVP scope** for the model/training side of the
project.

Its job is to answer:

> What must be true about the loader, model path, and training loop before we
> can trust that the model is actually training on the intended experiment?

This is a planning document, not a detailed implementation spec.

---

## MVP objective

Establish **one canonical training path** that:

- consumes the trusted processed data contract,
- uses one unambiguous loader path,
- trains the baseline multimodal model reproducibly,
- produces checkpoints/logs that correspond to the documented experiment.

---

## Current reality

What already exists:

- the multimodal model architecture is implemented,
- `src/train.py` provides a runnable Lightning-based training entrypoint,
- the project already has a parquet-based multi-home training flow,
- checkpointing and logging exist.

What is still weak:

- `src/dataloader.py` still contains stale CSV-era logic,
- data-loading truth is split across old and new paths,
- normalization is not integrated,
- reproducibility and contract boundaries are not yet fully documented,
- smoke-test validation exists but is not yet the final source of training truth.

---

## What must be addressed before writing the detailed implementation plan

### 1. Canonical loader decision

Before detailed planning, explicitly decide:

- the parquet multi-home path is the authoritative one,
- the stale CSV path is either retired or rewritten,
- there is one loader contract used by train/validate/eval.

### 2. Model scope for MVP

Lock what the MVP model is:

- keep the current architecture as the baseline,
- do not redesign fusion or encoders before the core experiment is stable,
- treat model novelty as secondary to training correctness.

### 3. Split policy for MVP

Document the exact training logic:

- source regions,
- target region,
- chronological validation policy,
- zero-shot target policy,
- what is and is not allowed to be seen during training.

### 4. Reproducibility contract

Need a clear policy for:

- config truth,
- seeds,
- checkpoint naming,
- log outputs,
- hardware-dependent behavior,
- required validation before claiming a run is valid.

### 5. Training smoke-test definition

Before detailed planning, define what counts as “the training pipeline works”:

- script runs end-to-end,
- batches are shaped correctly,
- loss decreases or is at least numerically stable,
- checkpoints are written,
- validation/test hooks execute,
- no stale codepath is involved.

---

## Global work packages

### Work package A — Canonical training codepath

- unify data loading,
- remove stale path ambiguity,
- ensure train/validate/eval all use the same contract,
- reduce duplication in data assumptions.

### Work package B — Data module and window contract

- define what one sample is,
- define how windows are created from processed data,
- define home-level and region-level boundaries,
- ensure the training path consumes normalized features correctly.

### Work package C — Training script hardening

- tighten config behavior,
- define run metadata and outputs,
- define checkpointing expectations,
- ensure training can be smoke-tested and reproduced.

### Work package D — Model baseline hardening

- treat the existing multimodal architecture as the MVP baseline,
- verify tensor contracts and modality assumptions,
- document what parts are baseline and what parts are later ablation targets.

### Work package E — Operational validation

- define smoke-test checks,
- define end-to-end validation expectations,
- define what evidence is required before saying training works.

---

## Deliverables for this workstream

1. one canonical training/data-loading path,
2. one documented train/val/test contract,
3. one stable training script for the MVP,
4. one smoke-test validation workflow,
5. one reproducibility checklist.

---

## Acceptance criteria

This workstream is ready when:

- only the intended loader path is authoritative,
- the model trains on the trusted processed data contract,
- the run configuration is reproducible and documented,
- checkpoints/logs are produced consistently,
- training/validation/test all reflect the documented experiment,
- the project can demonstrate that the model is actually training rather than
  merely running forward passes.

---

## What should not be over-scoped yet

Do **not** make this workstream responsible for:

- exhaustive hyperparameter search,
- advanced architecture redesign,
- domain adaptation methods,
- probabilistic forecasting,
- large-scale benchmark expansion.

---

## Consequent next step

After this global plan is accepted, the next document should be a **granular PDD
for the canonical model/data-loading/training path** under `docs/superpowers/`
with explicit task breakdown, file-level ownership, validation commands, and run
criteria.
