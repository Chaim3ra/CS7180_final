# MVP Plan 1 — Preprocessing, Normalization, and EDA

## Purpose

This document defines the **global MVP scope** for the data-preparation side of
 the project.

Its job is to answer one question:

> What must be true about the processed data before any training or evaluation
> claims are trustworthy?

This is **not** a line-by-line implementation plan. It is the higher-level plan
that should be settled before writing a detailed execution PDD.

---

## MVP objective

Produce one **canonical, trusted, reproducible processed dataset contract** for:

- TX source homes,
- CA source homes,
- NY zero-shot target homes.

That contract must be safe enough that later training and evaluation work can be
built on top of it without revisiting basic data-validity questions.

---

## Current reality

What already exists:

- `src/preprocess.py` aligns weather and solar and writes parquet outputs.
- `src/train.py` consumes processed parquets.
- `src/validate.py` checks basic pipeline assumptions.
- `data/README.md` documents the California data scarcity issue.

What is still weak:

- explicit timestamp sorting is not yet enforced,
- gap-safe windowing is not yet formalized,
- normalization is missing,
- EDA is not yet formalized as evidence,
- some metadata fields are proxies/fallbacks rather than verified site truth.

---

## What must be addressed before writing the detailed implementation plan

These decisions should be locked first.

### 1. Canonical cadence

- lock the project to **15-minute forecasting** for the MVP,
- define how hourly weather is aligned to 15-minute solar,
- explicitly reject mixing 1-minute and 15-minute assumptions inside the MVP.

### 2. Canonical weather source

- lock the MVP preprocessing path to the source currently used by code,
- document whether that is NASA POWER only for the MVP,
- defer weather-source comparisons until after the core pipeline works.

### 3. Canonical processed schema

Must be written down clearly:

- required columns,
- units,
- timestamp semantics,
- metadata semantics,
- allowed null policy,
- region/home identity fields.

### 4. Gap / continuity policy

Define what counts as a discontinuity and what to do about it:

- split sequences at gaps,
- drop invalid spans,
- prevent windows from crossing boundaries.

### 5. Normalization contract

Before detailed execution planning, decide:

- which columns are normalized,
- where scalers are fitted,
- where artifacts are stored,
- whether normalization happens in preprocessing or in the dataset layer.

### 6. EDA scope for the MVP

EDA should answer project-critical questions, not become open-ended exploration.

The MVP EDA should focus on:

- home counts and row counts by region,
- solar non-zero distribution,
- missingness/null patterns,
- timestamp continuity,
- weather value ranges,
- metadata completeness,
- target-region vs source-region distribution shift.

---

## Global work packages

### Work package A — Data contract and schema truth

- define the one trusted processed schema,
- define required columns and units,
- document train/source/target identities,
- make the processed parquet contract the single source of truth.

### Work package B — Timestamp and alignment correctness

- enforce timestamp parsing and sorting per home,
- verify cadence after alignment,
- define and implement discontinuity handling,
- validate weather-to-solar alignment assumptions.

### Work package C — Normalization and feature handling

- add train-only normalization,
- persist scaler artifacts,
- define transform application to source-val and NY test,
- document raw vs transformed feature spaces.

### Work package D — Data QA and validation

- range checks,
- null checks,
- duplicate checks,
- home-level quality filters,
- assertions that processed outputs are reproducible.

### Work package E — EDA and evidence generation

- create a compact EDA notebook/report,
- produce tables/plots that justify filtering and preprocessing,
- identify key region differences relevant to later evaluation.

---

## Deliverables for this workstream

The MVP output of this workstream should be:

1. one canonical processed parquet contract,
2. one normalization contract,
3. one data QA checklist,
4. one compact EDA artifact/report,
5. one written summary of known dataset limitations.

---

## Acceptance criteria

This workstream is ready when:

- every home sequence is explicitly sorted,
- no training window crosses a gap,
- normalization is leak-free and reproducible,
- processed files can be validated automatically,
- EDA explains the basic data shape and known risks,
- downstream training can rely on the processed schema without ambiguity.

---

## What should not be over-scoped yet

Do **not** make this workstream responsible for:

- architecture redesign,
- advanced transfer-learning methods,
- full explainability tooling,
- final paper-style result analysis,
- large weather-source comparison studies.

---

## Consequent next step

After this global plan is accepted, the next document should be a **granular PDD
for preprocessing/normalization execution** under `docs/superpowers/` that breaks
this workstream into concrete tasks, ownership boundaries, validation commands,
and file-by-file changes.
