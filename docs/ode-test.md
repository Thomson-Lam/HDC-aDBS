# ODE Testing Guide

This document covers how to test the STN-GPe ODE model and run the open-loop stimulation sanity gate.

## 1) Run core ODE/simulation tests

These tests validate deterministic seeding, sample-rate correctness, chunked runner parity, and signal safety checks.

```bash
uv run pytest tests/test_simulation.py -v
```

To run the full repository test suite:

```bash
uv run pytest tests -q
```

## 2) Run the open-loop stimulation sanity gate

This gate validates that stimulation has the expected directional effect on pathological beta behavior before closed-loop controller work.

### Command

```bash
uv run python ode-checks/open-loop-sanity.py
```

This runs a frozen 5 seed experiment (0, 1, 2, 3, 4) for 4 conditions per seed to validate behavior of the ODE model:
- healthy, no stimulation 
- pathological, no stimulation 
- pathological, weak open-loop stimulation 
- pathological, strong open-loop stimulation

Stimulations are applied using a fixed open loop pulse train (`make_pulse_train_stim`) at 130 Hz and a pulse width of 1ms, weak amplitude of 1.0 and strong amplitude of 8.0. 

Open loop means that pulses are delivered by schedule only and not via decisions from the feedback (in closed).

For each run, beta-band power (13-30 Hz) is computed from the LFP surrogate via Welch PSD (`beta_power_welch` in the code). We then compute the suppression ratios for each seed setting for the weak and strong pathological baselines to check model behavior and the effect of the stimulations (**not** classifier performance).

### What it runs (frozen lean gate)

- Seeds: `0,1,2,3,4`
- Conditions per seed:
  - healthy/no stimulation
  - pathological/no stimulation
  - pathological/weak stimulation
  - pathological/strong stimulation
- Stimulation: 130 Hz pulse train, 1 ms pulse width
- Metric: beta power (13-30 Hz) using Welch PSD integration

### Core checks

Gate passes only if all checks pass:

1. No runtime errors across all runs (`no_runtime_errors`)
2. Pathological beta > healthy beta in at least 4/5 seeds (`pathological_beta_gt_healthy_majority`)
3. Strong stimulation suppresses pathological beta in at least 4/5 seeds (`strong_stim_suppresses_pathological_majority`)
4. Strong stimulation supppresses more than weak stimulation on the dose response in at least 4/5 seeds (`dose_response_majority`)

## 3) Outputs 

After `ode-checks/open-loop-sanity.py` runs, it writes:

- `artifacts/open_loop_sanity/summary.yaml`
  - overall pass/fail
  - check booleans
  - aggregate ratio stats
  - any runtime errors
- `artifacts/open_loop_sanity/per_seed_metrics.csv`
  - per-seed beta values and suppression ratios
- `artifacts/open_loop_sanity/seed0_traces.png`
  - visual check of healthy/pathological/weak/strong traces for one seed
- `artifacts/open_loop_sanity/beta_summary.png`
  - mean +- std beta power by condition
- `artifacts/open_loop_sanity/open_loop_sanity.log`
  - single consolidated log file for the run

## 4) Fast smoke test for gate implementation

Use this to confirm gate plumbing and artifact writing in a short run:

```bash
uv run pytest tests/test_open_loop_sanity.py -v
```

This test uses a shortened configuration and checks that summary/plots/CSV are generated.
