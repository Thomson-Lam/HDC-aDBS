# HDC Models: Low-level Implementation Guide

This document explains the current HDC model stack in code-level terms: where each component lives, what each function/class does, and how validator search and training are wired together.

## 1) Scope

This guide covers the implemented HDC model components:

- hypervector primitives,
- dictionary initialization,
- window encoder,
- readout variants (prototype and linear),
- trainer layer,
- validator search and ranking,
- train/validation orchestration script.

It does not define the future ODE-sampling dataset/split integration layer.

## 2) Code Defs 

- `hdc/primitives.py`
  - `bipolarize(x, zero_to=1)`
  - `bind(a, b)`
  - `bundle(hypervectors, axis=0, zero_to=1)`
  - `l2_normalize(x, axis=-1, eps=1e-12)`
  - `normalized_dot(a, b, eps=1e-12)`
- `hdc/initializers.py`
  - `BaseHypervectorInitializer`
  - `RandomBinaryInitializer`
  - `RFFBinaryInitializer`
- `hdc/dictionaries.py`
  - `DictionaryConfig`
  - `build_value_dictionary(...)`
  - `build_position_dictionary(...)`
  - `build_dictionaries(config)`
- `hdc/encoder.py`
  - `EncoderConfig`
  - `WindowEncoder`
    - `zscore_window(window)`
    - `quantize(z_window)`
    - `encode_window(window)`
    - `encode_batch(windows)`
- `hdc/readouts.py`
  - `BaseReadout`
  - `PrototypeReadout`
  - `LinearReadout`
- `hdc/training.py`
  - `BaseHDCTrainer`
  - `PrototypeHDCTrainer`
  - `LinearHDCTrainer`
- `hdc/search/config.py`
  - `EncoderSearchSpec`
  - `SearchConfig`
- `hdc/search/validator.py`
  - `SplitData`, `ValidationData`, `SearchResult`
  - `run_validator_search(...)`
  - `rank_results(...)`
  - `top_candidates(...)`
  - `write_results_jsonl(...)`, `write_results_csv(...)`
- `hdc/search/run.py`
  - `make_dummy_validation_data(...)`
  - `main()` fixed-grid search runner
- `train/valid-train.py`
  - `ensure_freeze_record(...)`
  - `train_both_methods(...)`
  - `main()`

## 3) Data contracts

- Label convention is fixed across stack:
  - `0`: healthy
  - `1`: pathological
- Window input contract:
  - rank-2 array for batch APIs: `(n_windows, window_length)`
  - current default `window_length=128` (512 ms at 250 Hz)
- Encoded hypervector contract:
  - shape: `(n_windows, D)`
  - dtype: `int8`
  - values: bipolar `{-1, +1}`

## 4) Hypervector primitives and dictionaries

### Primitives

- `bind(a, b)` uses elementwise sign-product for bipolar binding.
- `bundle(...)` performs majority-style bundling as sum + bipolarization.
- `bipolarize(...)` defines tie behavior explicitly via `zero_to`.
- `normalized_dot(...)` computes cosine-like similarity via normalized inner product.

### Dictionary initialization

- Value dictionary (`n_bins x D`) can use:
  - random binary (`RandomBinaryInitializer`), or
  - RFF correlated initializer (`RFFBinaryInitializer`).
- Position dictionary (`window_length x D`) is random binary for MVP.
- `build_dictionaries(...)` uses deterministic seed offsets so value and position codebooks are reproducible.

## 5) Encoder internals

`WindowEncoder` executes the fixed pipeline:

1. Per-window z-score normalization (`zscore_window`).
2. Clipped quantization to bin ids (`quantize`) using fixed `clip_z`.
3. Value-position binding with dictionary lookup.
4. Bundling across sample positions.

One input window yields one bipolar hypervector.

## 6) Readout variants

### Prototype-similarity readout (`PrototypeReadout`)

- `fit(X_hv, y)`:
  - bundle healthy HVs -> healthy prototype,
  - bundle pathological HVs -> pathological prototype.
- `decision_function(X_hv)`:
  - `sim_path - sim_healthy` using `normalized_dot`.
- `predict(...)`:
  - threshold margin at 0 by default.

### Linear readout (`LinearReadout`)

- `fit(X_hv, y)` trains logistic regression (`liblinear`).
- `decision_function(X_hv)` returns linear margin.
- `predict(...)` thresholds margin at 0 by default.

Both readouts share one prediction interface through `BaseReadout`.

## 7) Trainer layer

The trainer layer wraps encoder + readout so raw windows can be used directly.

### `BaseHDCTrainer`

- shared methods:
  - `encode(windows)`
  - `predict(windows, threshold=0.0)`
  - `evaluate(windows, y)`
- abstract methods:
  - `fit(...)`
  - `decision_function(...)`
  - `save(...)` / `load(...)`

### `PrototypeHDCTrainer`

- uses `WindowEncoder` + `PrototypeReadout`.
- saves:
  - `metadata.json`
  - `encoder_dictionaries.npz`
  - `readout.npz` with `healthy_prototype`, `path_prototype`.

### `LinearHDCTrainer`

- uses `WindowEncoder` + `LinearReadout`.
- saves:
  - `metadata.json` (includes linear metadata)
  - `encoder_dictionaries.npz`
  - `readout.npz` with `coef`, `intercept`, `classes`, and model shape metadata.

## 8) Validator search structure

`SearchConfig` in `hdc/search/config.py` freezes the MVP search space:

- dimensions: `1000, 5000, 10000`
- bins: `8, 16`
- value initializers: `random, rff`
- readouts: `prototype, linear`
- window and stride contracts are fixed.

`run_validator_search(...)` in `hdc/search/validator.py` does:

1. iterate encoder candidates,
2. encode train/validation windows,
3. fit both readouts,
4. compute metrics,
5. rank by frozen rule.

Ranking rule:

- primary: balanced accuracy on clean validation,
- guardrail: minimum AUROC over onset and recovery validation,
- tie-breaker: mean encoding time per window.

## 9) `train/valid-train.py` orchestration

Current script flow:

1. Load freeze record if present (`artifacts/encoder_search/freeze_record.yaml`).
2. If missing, run validator search and create freeze record + leaderboard outputs.
3. Build frozen `EncoderConfig`.
4. Train both `PrototypeHDCTrainer` and `LinearHDCTrainer`.
5. Evaluate on clean validation split.
6. Save model artifacts and write `artifacts/models/train_report.yaml`.
7. Load models back and run score-delta check for serialization correctness.

Important current behavior:

- `train/valid-train.py` uses dummy data from `hdc/search/run.py`.
- This is intentional for now and should be replaced by real trajectory split adapters once the ODE data pipeline is available.

## 10) Test coverage relevant to models

- `tests/test_hdc_core.py`
  - primitives, encoder invariants, readout behavior.
- `tests/test_reproducibility_dummy.py`
  - deterministic encoding/readout reproducibility checks.
- `tests/test_validator_search.py`
  - fixed search-space coverage and ranking behavior.
- `tests/test_training_layer.py`
  - trainer fit/evaluate/save/load roundtrip checks.

## 11) Quick commands

Run all tests:

```bash
uv run python -m unittest discover -s tests -v
```

Run validator + training pipeline:

```bash
uv run python train/valid-train.py
```
