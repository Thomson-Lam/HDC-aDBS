# HDC Encoder Validation and Evaluation

This document defines how the HDC encoder is validated offline, how hypervectors are evaluated before controller lock-in, and how the frozen detector is evaluated in the closed loop.

It follows the global experiment contract in `docs/experiment-spec.md` and uses the detailed encoder search options listed in `docs/hdc-encoder-optsearch.md`.

## 1) Scope and intent

The goal is to choose one sane encoder/readout configuration for the MVP without leaking test information and without blending optimization with final evaluation.

The process is staged:

1. offline encoder validation and ranking on hold-out validation data,
2. freeze one selected configuration,
3. run final offline held-out reporting and final closed-loop comparison on untouched test trajectories.

## 2) Fixed signal and controller contract

All offline and online HDC evaluations use the same controller-visible contract as per `experiment-spec.md`:

- resampled stream at 250 Hz,
- decision cadence and window stride at 50 ms,
- window length 512 ms (128 samples),
- per-window z-score normalization,
- causal online processing,
- same shared input stream used by both HDC and classical beta-threshold baselines.

## 3) Encoder search space

Check [Encoder Search Space](encoder-searchspace.md).

## 4) Data split policy and leakage control

Data is split by trajectory/run/seed before window extraction.

- `train`: used to fit encoder-dependent artifacts and train readouts,
- `validation`: used for encoder ranking, threshold calibration, and model-selection decisions,
- `test`: untouched until all settings are frozen.

Overlapping windows may exist within a split but may never cross split boundaries.

Healthy-only holdout trajectories are reserved for specificity checks (false-trigger behavior and unnecessary stimulation).

## 5) Offline validator process

For each candidate encoder configuration:

1. fit prototypes/readout using `train` only,
2. evaluate on hold-out `validation` windows,
3. compute clean-state discrimination, transitional robustness, and encoding latency,
4. log results with fixed metric definitions and deterministic seeds where applicable.

Readout comparison (prototype similarity vs linear readout over hypervectors) is part of this offline validator and is evaluated under the same split and ranking policy.

## 6) Frozen ranking rule

Encoder ranking is frozen before test and closed-loop comparison:

- primary metric: balanced accuracy on clean validation windows (healthy vs pathological),
- guardrail: minimum AUROC on onset and recovery validation windows,
- tie-breaker: mean encoding time per window.

Any candidate failing the guardrail is ineligible even if its primary score is high.

## 7) Selection funnel and freeze point

To keep efficiency while preserving rigor:

1. run the full offline validator over the candidate set,
2. keep the top 2-3 validation-ranked candidates,
3. run a small validation-only controller sweep on those candidates,
4. freeze one final encoder/readout/threshold-smoothing-hysteresis setting,
5. only then run final held-out test and final closed-loop comparison.

No test-set performance is used for ranking or threshold setting.

## 8) Offline held-out evaluation after freeze

After freezing, report test-set offline results once:

- clean healthy vs pathological discrimination,
- onset/recovery/moderate-coupling robustness,
- healthy false-trigger behavior,
- per-window decision time and memory footprint.

This report is descriptive and confirmatory, not a new optimization pass.

## 9) In-loop (closed-loop) evaluation with frozen HDC

The frozen HDC detector is integrated into the matched controller state machine and compared under the same fairness constraints as the classical baseline.

Core conditions:

1. no stimulation,
2. continuous DBS,
3. classical beta-threshold aDBS,
4. HDC-triggered aDBS.

Core reported metrics:

- mean beta power,
- pathological occupancy,
- suppression latency,
- duty cycle and pulse count,
- healthy false-trigger rate,
- decision time per window and memory use.

## 10) Non-negotiable rules

- Do not tune encoder family or ranking policy on test data.
- Do not change the signal contract between offline and online phases.
- Do not alter thresholding/smoothing/hysteresis/lockout after freeze.
- Document any deviations as an explicit spec revision before reruns.
