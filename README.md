# HDC-aDBS Testbed

Lightweight Hyperdimensional Computing (HDC) control for adaptive deep brain stimulation (aDBS) in an in silico Parkinsonian STN-GPe model.

This repo is currently in the **simulation + offline HDC core** stage.

## What This Project Is

- Simulate healthy and pathological neural dynamics with an STN-GPe ODE model.
- Derive a shared LFP-like surrogate stream for controller input.
- Evaluate HDC-based state detection against a classical beta-threshold baseline.
- Progress toward a fair 4-condition closed-loop comparison:
  - no stimulation,
  - continuous DBS,
  - classical beta-threshold aDBS,
  - HDC-triggered aDBS.

## Backlog

- [x] 1) Lock experiment rules (spec table and fixed contract)
- [x] 2) Build simulation and signal pipeline

- [ ] 3) Run open-loop stimulation sanity gate (formal scripted gate pending)

> TODO: run 3) ! 

- [x] 4) Build HDC core library
- [x] 4.5) Build HDC models that call the primitives

- [ ] 5) Build dataset + split layer (core implemented; transitional subsets pending)

- [x] 6) Build offline validator + ranking engine (currently wired to dummy data)

- [ ] 7) Run selection funnel + freeze final HDC setup (partial: freeze record exists)
- [ ] 8) Calibrate thresholds + held-out offline report
- [ ] 9) Integrate and evaluate closed-loop controllers
- [ ] 10) Add full reproducibility + coverage for post-dataset stages

## Implemented Components Checklist

### Simulation and signal path

- [x] STN-GPe ODE right-hand side (`src/simulation/model.py`)
- [x] Single-run and chunked runner (`src/simulation/runner.py`)
- [x] LFP surrogate extraction (`src/simulation/lfp.py`)
- [x] Variable-step -> fixed 250 Hz resampling
- [x] Signal safety checks (finite values, monotonic timestamps, discontinuity checks)
- [x] Deterministic seeding support

### HDC core stack

- [x] Bipolar hypervector primitives (`hdc/primitives.py`)
- [x] Random + RFF dictionary initialization (`hdc/initializers.py`)
- [x] Window encoder pipeline (`hdc/encoder.py`)
- [x] Prototype and linear readouts (`hdc/readouts.py`)
- [x] Trainer layer with save/load (`hdc/training.py`)

### Offline validator / training flow

- [x] Fixed search config (`hdc/search/config.py`)
- [x] Frozen ranking rule with guardrail (`hdc/search/validator.py`)
- [x] Search artifact outputs (`results.jsonl`, `leaderboard.csv`)
- [x] Freeze record generation (`artifacts/encoder_search/freeze_record.yaml`)
- [x] Combined validator + train script (`train/valid-train.py`)
- [ ] ODE-backed dataset plugged into validator/train (currently dummy provider)

### Testing

- [x] Simulation tests (`tests/test_simulation.py`)
- [x] HDC core tests (`tests/test_hdc_core.py`)
- [x] Validator ranking tests (`tests/test_validator_search.py`)
- [x] Training save/load tests (`tests/test_training_layer.py`)
- [x] Reproducibility checks (`tests/test_reproducibility_dummy.py`)

## What Is Still Missing (Short List)

- [ ] Trajectory-level dataset/split module (healthy/pathological/onset/recovery/moderate/healthy-only holdout)
- [ ] Leakage guards for overlapping windows across splits
- [ ] Open-loop stimulation sanity gate script with pass/fail report
- [ ] Classical and HDC closed-loop controller state machines with matched mechanics
- [ ] Final threshold calibration on validation-only and single held-out test report

## Quickstart

1. Install dependencies:

```bash
uv sync
```

2. Run tests:

```bash
uv run pytest tests -q
```

3. Run validator + training flow:

```bash
uv run python train/valid-train.py
```

4. Build and split static ODE dataset:

```bash
uv run python train/build-static-dataset.py
uv run python train/prepare-static-splits.py
```

## Current Training Flow (Important)

- `train/valid-train.py` currently uses synthetic dummy windows for end-to-end plumbing.
- This is intentional while the ODE-backed dataset/split integration is being completed.

## Produced results 

- Encoder search:
  - `artifacts/encoder_search/results.jsonl`
  - `artifacts/encoder_search/leaderboard.csv`
  - `artifacts/encoder_search/freeze_record.yaml`
- Trained models:
  - `artifacts/models/prototype/`
  - `artifacts/models/linear/`
  - `artifacts/models/train_report.yaml`

## Repository Layout

```text
brain/
├── README.md
├── configs/
│   └── sim_config.py
├── src/
│   └── simulation/
│       ├── model.py
│       ├── runner.py
│       └── lfp.py
├── hdc/
│   ├── primitives.py
│   ├── initializers.py
│   ├── dictionaries.py
│   ├── encoder.py
│   ├── readouts.py
│   ├── training.py
│   └── search/
│       ├── config.py
│       ├── validator.py
│       └── run.py
├── train/
│   └── valid-train.py
├── tests/
│   ├── test_simulation.py
│   ├── test_hdc_core.py
│   ├── test_validator_search.py
│   ├── test_training_layer.py
│   └── test_reproducibility_dummy.py
└── docs/
    ├── experiment-spec.md
    ├── encoder-searchspace.md
    ├── models.md
    └── planning/
```

## Notes

- Locked experiment contract lives in `docs/experiment-spec.md`.
- This is a research prototype (not a clinical system).
