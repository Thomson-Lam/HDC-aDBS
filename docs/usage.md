# End-to-End Usage

This guide covers the full flow:

1. build the static ODE data pipeline,
2. train and freeze HDC models,
3. inspect held-out test metrics,
4. run the four-condition closed-loop benchmark and export metrics.

## Prerequisites

- From repo root: `brain/`
- Install dependencies once:

```bash
uv sync
```

## Quick Path (Minimal Commands)

If you just want the full pipeline with default settings:

```bash
uv run python train/build-static-dataset.py
uv run python train/prepare-static-splits.py
uv run python train/valid-train.py
uv run python -m controllers.run_closedloop_benchmark
```

This produces training artifacts under `artifacts/models/` and closed-loop metrics under `artifacts/closed_loop/`.

## Step-by-Step

### 1) Build static trajectories

```bash
uv run python train/build-static-dataset.py
```

Outputs:

- `artifacts/datasets/static_v1/trajectories/*.npz`: individual trajectories as np arrays
- `artifacts/datasets/static_v1/manifest.csv`: index/metadata table pointing to np arrays
- `artifacts/datasets/static_v1/build_config.yaml`: configs used to generate stored for reproducibility

Our outputs:

```
uv run python train/build-static-dataset.py
Built static dataset: artifacts/datasets/static_v1
Manifest: artifacts/datasets/static_v1/manifest.csv
Build config: artifacts/datasets/static_v1/build_config.yaml
```

### 2) Vet data and assign train/val/test/holdout splits

```bash
uv run python train/prepare-static-splits.py
```

Outputs:

- `artifacts/datasets/static_v1/manifest_with_splits.csv`: original manifest.csv from step 1 with split column processed
- `artifacts/datasets/static_v1/split_scenario_counts.csv`: summary csv of how many trajectories of each scenario landed in each split
- `artifacts/datasets/static_v1/window_subset_counts.csv`: summary of window level counts (using `window_length=128, `stride_ms=50` for default vals) and reports counts by split and subsets (`clean`, `onset`, `recovery`, `moderate`, `context`, `healthy_holdout`)
- `artifacts/datasets/static_v1/vet_issues.csv` (only if issues exist)

Our outputs:

```
uv run python train/prepare-static-splits.py
Valid trajectories: 30
Issue rows: 0
Split manifest: artifacts/datasets/static_v1/manifest_with_splits.csv
Split scenario counts: artifacts/datasets/static_v1/split_scenario_counts.csv
Window subset counts: artifacts/datasets/static_v1/window_subset_counts.csv
split
train      15
val         5
test        5
holdout     5
```

### 3) Run validator search and train models

```bash
uv run python train/valid-train.py
```

What this does:

- runs encoder search if no freeze record exists,
- freezes the selected encoder config,
- trains both `prototype` and `linear` readouts,
- auto-selects linear regularization (`C`) on clean validation,
- calibrates decision thresholds on clean validation with an FPR target,
- reports validation and held-out test metrics,
- writes reproducibility audit outputs.

Key outputs:

- `artifacts/encoder_search/freeze_record.yaml`: frozen winner config + key selection metrics so training/eval can be reproduced exactly.
- `artifacts/encoder_search/leaderboard.csv`: ranked results of all searched encoder/readout candidates with their validation/guardrail metrics.
- `artifacts/models/prototype/`: serialized prototype-HDC model artifacts ready for load/inference in offline or closed-loop runs.
- `artifacts/models/linear/`: serialized linear-readout-over-HDC model artifacts ready for load/inference in offline or closed-loop runs.
- `artifacts/models/train_report.yaml`: compact training summary with validation, held-out test metrics, and healthy-holdout false-trigger results.
- `artifacts/models/training_audit.yaml`: full audit trail (data coverage, guardrail thresholds, selected config, and reported metrics) for provenance/reproducibility.

Our outputs:
```
uv run python train/valid-train.py
Training complete
Prototype clean metrics: {'balanced_accuracy': 0.6190476190476191, 'auroc': 0.6057256235827664}
Linear clean metrics: {'balanced_accuracy': 0.44047619047619047, 'auroc': 0.49603174603174605}
Prototype load-check max score delta: 0.000000000000
Linear load-check max score delta: 0.000000000000
Freeze record: artifacts/encoder_search/freeze_record.yaml
Train report: artifacts/models/train_report.yaml
Training audit: artifacts/models/training_audit.yaml
```

### 4) Inspect offline test-set metrics

The easiest source is:

- `artifacts/models/train_report.yaml`

Look at:

- `prototype_test`
- `linear_test`

These are held-out split metrics reported after training.

If you want notebook exploration, open `notebooks/01_sim_validation.ipynb` and load artifacts from `artifacts/models/` and `artifacts/datasets/static_v1/`.

### 4.5) Run calibration + overfitting ablations (Phases 1-4)

This runs threshold calibration, train-vs-val diagnostics, normalization ablations, and linear regularization sweeps.

```bash
uv run python train/calibrate-overfitting.py
```

Outputs:

- `artifacts/overfit_calibration/phase1_4_experiments.csv`
- `artifacts/overfit_calibration/phase1_4_summary.yaml`

Notes:

- This script uses existing static dataset + split artifacts.
- It does not rebuild trajectories unless your dataset artifacts are missing.

### 5) Run closed-loop benchmark (4 conditions)

```bash
uv run python -m controllers.run_closedloop_benchmark
```

Default conditions:

1. `no_stimulation`
2. `continuous_dbs` (pulse train always on)
3. `beta_adbs` (pulse-train epochs)
4. `hdc_adbs` (pulse-train epochs)

Default output files:

- `artifacts/closed_loop/per_run_metrics.csv`
- `artifacts/closed_loop/summary_by_condition.csv`
- `artifacts/closed_loop/summary.yaml`
- `artifacts/closed_loop/run_manifest.yaml`

### 6) Generate report plots from artifacts

Generate individual images for search, offline training, calibration, overfitting diagnostics, and (if present) closed-loop summaries:

```bash
uv run python plots/generate_all.py
```

Default output directory:

- `artifacts/plots/`

If closed-loop artifacts are missing, the closed-loop plot scripts are skipped automatically.

Our outputs:

```
 uv run python -m controllers.run_closedloop_benchmark
{
  "per_run_metrics_csv": "artifacts/closed_loop/per_run_metrics.csv",
  "summary_by_condition_csv": "artifacts/closed_loop/summary_by_condition.csv",
  "run_manifest_yaml": "artifacts/closed_loop/run_manifest.yaml"
}
```

## Closed-Loop Command Options

Use this when you want explicit seeds, model type, and output path:

```bash
uv run python -m controllers.run_closedloop_benchmark \
  --model-path artifacts/models/linear \
  --trainer-type linear \
  --seeds 0,1,2,3,4 \
  --healthy-seeds 100,101,102,103,104 \
  --output-dir artifacts/closed_loop
```

### Flags:

- `--model-path` path to trained model directory (`artifacts/models/linear` or `artifacts/models/prototype`)
- `--trainer-type` one of `linear` or `prototype`
- `--seeds` pathological-run seeds (comma-separated)
- `--healthy-seeds` healthy holdout seeds (comma-separated)
- `--beta-threshold` beta controller threshold
- `--hdc-threshold` HDC controller threshold
- `--stim-amplitude` stimulation amplitude
- `--t-end-ms` and `--t-warmup-ms` simulation duration settings

## Recommended Evaluation Order

For final reporting quality:

1. run data build + split,
2. train/freeze models,
3. calibrate thresholds on validation-only windows,
4. confirm held-out offline test metrics,
5. run closed-loop benchmark and report CSV summaries.

## Tests

- Run full tests:

```bash
uv run pytest tests -q
```

- Run only closed-loop tests:

```bash
uv run python -m pytest tests/test_controllers.py -q
uv run python -m pytest tests/test_closedloop_metrics.py -q
```
