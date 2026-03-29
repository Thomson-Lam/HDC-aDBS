# Closed-Loop HDC Controller — Design Document

## Motivation

The HDC stack (encoder, readouts, training) and ODE simulator are complete.
This document covers the design of the closed-loop feedback layer that connects
them: an HDC-based controller that detects pathological beta bursts in real time
and injects stimulation current into the STN model.

A classical beta-power baseline controller is also included so that the two
approaches can be compared under identical simulation conditions.

---

## Architecture Overview

```
SimConfig (regime, timing)
        │
        ▼
run_closed_loop()          ← controllers/run_controller.py
  ├── for each 50 ms chunk:
  │     ├── controller.get_stim_fn()   → stim_fn(t) → float
  │     ├── solve_ivp(rhs, [t_a, t_b], y, stim_fn)
  │     ├── resample_to_fs()           → 250 Hz grid
  │     ├── extract_lfp()              → mean STN voltage
  │     └── controller.ingest(t, lfp) → updates buffer + decisions
  └── returns {t, lfp, stim, state, metrics}

BaseController             ← controllers/base.py
  ├── Ring buffer (128 samples, circular)
  ├── State machine (IDLE → STIMULATING → LOCKOUT → IDLE)
  ├── Decision cadence (every 12–13 samples = 50 ms)
  └── _compute_decision(window) [abstract]
        │
        ├── HDCController  ← controllers/hdc_controller.py
        │     └── trainer.decision_function(window[None, :])
        │
        └── BetaController ← controllers/beta_controller.py
              └── RMS of causal bandpass-filtered window (13–30 Hz)
```

---

## Experiment Spec (Locked)

All values from `docs/experiment-spec.md`:

| Parameter             | Value     | Notes                                      |
|-----------------------|-----------|--------------------------------------------|
| Sampling rate         | 250 Hz    | Fixed; matches SimConfig.fs                |
| Decision cadence      | 50 ms     | = chunk_duration_s in run_closed_loop      |
| Window length         | 128 samples (512 ms) | Captures full beta cycle        |
| Stim burst duration   | 200 ms    | Fixed pulse length                         |
| Post-stim lockout     | 200 ms    | Refractory period; prevents chatter        |
| Filtering policy      | Causal only | No non-causal filtering in controller   |
| Normalisation         | Per-window z-score (inside encoder) | Applied in HDC path |

---

## File Map

| File                            | Purpose                                              |
|---------------------------------|------------------------------------------------------|
| `controllers/base.py`           | `BaseController`, `ControllerConfig`, `ControllerMetrics`, `StimState` |
| `controllers/hdc_controller.py` | `HDCController` — wraps trained HDC model            |
| `controllers/beta_controller.py`| `BetaController` — causal bandpass + RMS             |
| `controllers/run_controller.py` | `run_closed_loop()` — simulation harness             |
| `controllers/__init__.py`       | Public exports                                       |
| `tests/test_controllers.py`     | Unit + integration tests                             |

---

## Ring Buffer

A fixed-size circular buffer of `window_length` (128) float64 samples.

```
_buffer[0..127]    — preallocated array
_buffer_head       — next write index (wraps at 128)
_buffer_fill       — 0..128, saturates once full
```

`get_latest_window()` reconstructs the chronological window by rotating from
`_buffer_head` (oldest sample) forward by `window_length` steps.

---

## State Machine

```
                    score >= threshold
                    AND state == IDLE
                    AND is_ready
         ┌──────────────────────────────┐
         │                              │
         ▼                              │
      IDLE  ──── trigger ────►  STIMULATING  ──── t >= stim_end_t ────►  LOCKOUT
         ▲                                                                    │
         └────────────────── t >= lockout_end_t ───────────────────────────┘
```

- `_stim_end_t = trigger_t + stim_duration_ms`
- `_lockout_end_t = _stim_end_t + lockout_duration_ms`
- Detections during STIMULATING / LOCKOUT are counted but do not retrigger.

---

## Decision Cadence

At 250 Hz, 50 ms = 12.5 samples.  `_decision_stride = round(12.5) = 12` (Python's
`round()` uses banker's rounding; `round(12.5) = 12`).  The per-sample counter
`_samples_since_decision` ensures at most one decision per stride.

---

## HDCController Decision Path

```
raw window (128 samples, mV)
    → trainer.decision_function(window[None, :])
        → WindowEncoder.zscore_window()    (per-window z-score, inside encoder)
        → quantize() → bind() → bundle()   (HDC encoding)
        → readout.decision_function()      (prototype diff or logistic margin)
    → scalar float (positive = pathological)
    → compared against config.threshold (default 0.0)
```

The z-score is applied inside `WindowEncoder`, so raw mV is passed here.

---

## BetaController Filter

2nd-order Butterworth bandpass (13–30 Hz) implemented via `scipy.signal.butter`
in SOS form.  The IIR filter state `_zi` is carried across chunk boundaries so
the filter is globally causal.  `BetaController` overrides `ingest()` to apply
the filter to each incoming chunk before pushing samples into the ring buffer.

Decision score = RMS of the filtered window in mV.

---

## Harness: run_closed_loop()

Replicates the inner loop of `src/simulation/runner.run_chunked()` with one
key difference: `stim_fn` is rebuilt from `controller.get_stim_fn()` at each
chunk boundary.

Grid-alignment math (from `runner.py:382-384`) ensures consecutive 50 ms chunks
tile the global 250 Hz grid without gaps:

```python
n_first      = ceil((t_a - t_start) / dt_ms)
n_last       = ceil((t_b - t_start) / dt_ms) - 1
t_chunk_grid = t_start + arange(n_first, n_last + 1) * dt_ms
```

The raw ODE end-state `sol.y[:, -1]` is carried to the next chunk (not the
interpolated state) to avoid accumulating resampling error.

---

## Threshold Calibration (Backlog Item 8)

`ControllerConfig.threshold` defaults to `0.0` (natural HDC margin zero-crossing).
Future calibration will set this from held-out healthy data to achieve a target
false-positive rate.  The `threshold` field is exposed on `ControllerConfig` for
easy override without changing any other code.

---

## Verification

```bash
# Unit tests (fast, no ODE):
pytest tests/test_controllers.py -v -k "not slow"

# Integration tests (ODE in the loop, ~10–30 s):
pytest tests/test_controllers.py -v -m slow

# Smoke test with real HDC model:
python - <<'EOF'
from configs.sim_config import pathological_config
from controllers import HDCController, ControllerConfig, run_closed_loop
ctrl   = HDCController("artifacts/models/linear", ControllerConfig(threshold=0.0))
result = run_closed_loop(ctrl, pathological_config(), seed=42)
print(result['metrics'].to_dict())
EOF
```
