# Harden Closed-Loop Plan (Option B: Pulse-Train Epochs)

## Purpose

Harden closed-loop evaluation so final claims are credible and reproducible.

This plan explicitly adopts **Option B** for stimulation waveform policy:

- controllers decide **when** stimulation epochs start/stop,
- stimulation delivered during each epoch is a **130 Hz pulse train** with fixed pulse width,
- waveform parameters are shared across conditions for fairness.

## Why Option B

- Better matches DBS-style stimulation assumptions in project docs.
- Enables pulse-based metrics (pulse count, pulses per second).
- Keeps controller comparison fair by changing only the decision statistic (beta vs HDC), not waveform form.

## Frozen Waveform Contract

Use one waveform config across all adaptive and continuous conditions:

- pulse frequency: `130 Hz`
- pulse width: `1.0 ms`
- pulse amplitude: fixed per experiment setting
- stim epoch duration: `200 ms`
- post-stim lockout: `200 ms`
- decision cadence: `50 ms`

No per-controller waveform tuning.

## Implementation Plan

## 1) Add pulse-train stim primitive for controllers

Create a shared utility in `controllers/` (or `src/simulation/`) that returns
`stim_fn(t_ms)` for pulse-train epochs.

Requirements:

- accepts epoch active/inactive state from controller
- when active: outputs pulse train (amp during pulse width, else 0)
- when inactive: outputs 0

## 2) Update controller state-machine output to drive waveform gating

In `controllers/base.py`:

- keep existing state transitions (`IDLE -> STIMULATING -> LOCKOUT`)
- expose an "epoch active" boolean based on state/time
- route stimulation through pulse-train primitive during `STIMULATING`

Result:

- controller still decides epoch timing,
- waveform inside epoch is now pulse train.

## 3) Align continuous DBS baseline with same pulse waveform

In closed-loop benchmark runner:

- implement continuous DBS as pulse-train always active
- do not use a different waveform type for continuous condition

This keeps comparisons focused on control policy, not waveform mismatch.

## 4) Extend runtime metrics in controller core

In `ControllerMetrics` (`controllers/base.py`), add:

- `pulse_count`
- `stim_onset_times_ms`
- `blocked_detections_lockout`
- `blocked_detections_stimulating`
- `decision_times_ms` summary stats (mean/p95)

Keep existing:

- `n_decisions`, `n_detections`, `n_stimulations`, `duty_cycle`.

## 5) Add closed-loop metric evaluator module

Add `controllers/eval_metrics.py` with pure functions:

- mean beta power (13-30 Hz)
- pathological occupancy
- suppression latency
- duty cycle
- pulse count
- healthy false-trigger rate
- decision-time stats
- memory stats (lightweight sampling)

All metric definitions should be locked in `docs/experiment-spec.md` language.

## 6) Build one four-condition benchmark script

Add `controllers/run_closedloop_benchmark.py`:

Conditions:

1. no stimulation
2. continuous DBS (pulse train always on)
3. beta-threshold aDBS (pulse-train epochs)
4. HDC-triggered aDBS (pulse-train epochs)

Outputs:

- `artifacts/closed_loop/per_run_metrics.csv`
- `artifacts/closed_loop/summary_by_condition.csv`
- `artifacts/closed_loop/summary.yaml`
- `artifacts/closed_loop/run_manifest.yaml`

## 7) Add healthy-specificity evaluation

Run healthy-holdout trajectories for adaptive controllers and report:

- false-trigger rate
- unnecessary stimulation duty cycle
- pulses per minute on healthy runs

## 8) Add tests

Add `tests/test_closedloop_metrics.py` and extend `tests/test_controllers.py`:

- pulse waveform correctness during epoch
- state-machine timing still correct with pulse output
- pulse counting correctness
- metric correctness on synthetic traces
- fast integration smoke for 4-condition runner

Keep heavy multi-seed benchmark as opt-in slow test.

## 9) Reproducibility and audit output

For each benchmark run write:

- config snapshot
- seeds
- git commit (if available)
- timestamps
- metric definitions version

Store under `artifacts/closed_loop/run_manifest.yaml`.

## Acceptance Criteria

This hardening phase is complete when:

- all stimulation conditions use pulse-train waveform policy (Option B)
- four-condition benchmark runs from one command
- required metric suite is present in output artifacts
- healthy-specificity metrics are included
- tests cover waveform, metric math, and state-machine invariants
- reruns with fixed seeds/config produce consistent summary outputs

## Suggested Execution Order

1. waveform primitive + controller wiring
2. metric module
3. benchmark runner
4. artifact writers
5. tests
6. first hardened benchmark run and artifact review
