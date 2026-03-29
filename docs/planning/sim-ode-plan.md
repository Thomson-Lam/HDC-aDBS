# Plan: ODE Simulator Component (Backlog Phase 2)

## Context
Building the simulation and signal pipeline — the first executable component of the testbed. Without it, nothing downstream (HDC encoding, controllers, comparison) can run. The goal is to produce clean healthy and pathological LFP signals, resampled to exactly 250 Hz, with verified beta-band separation between regimes.

This plan strictly follows the specs in `docs/planning/testbed-impl.md` Phase 1 and `docs/planning/backlog.md` Phase 2.

---

## Files to Create

| File | Role | Status |
|---|---|---|
| `configs/sim_config.py` | ODE parameters, regime definitions, seed config | ✅ Done |
| `src/simulation/model.py` | STN-GPe ODE function (right-hand side) | ✅ Done |
| `src/simulation/runner.py` | Chunked simulator + resampler + safety checks | ✅ Done |
| `src/simulation/lfp.py` | LFP surrogate extraction from model state | ✅ Done |
| `src/simulation/__init__.py` | Package re-exports | ✅ Done |
| `notebooks/01_sim_validation.ipynb` | Verification notebook (PSD plots, regime check) | ✅ Done |
| `tests/test_simulation.py` | Unit tests for correctness invariants | ✅ Done |

No `dynamics.py` — that split is unnecessary for the MVP 2-population model; keep model logic in `model.py`.

---

## Population Upgrade: 1+1 → 8+8

### Why

The 1+1 single-cell topology failed the sanity gate:
- Pathological beta power was **0.67× healthy** (reversed — should be > 1×)
- The published Terman-Rubin parameter values (`g_GABA=1.5`) assume a 16-cell population where 16 simultaneous GPe→STN projections deliver sufficient total inhibition to trigger the rebound synchrony loop
- With a single connection the effective inhibitory drive is ~16× too weak, so STN never enters the beta entrainment regime
- Seeds were unstable across regime checks (seed 1 was borderline), confirming the single-cell topology is too sensitive to initial conditions

### Changes required

#### `configs/sim_config.py`
- Change defaults: `n_STN = 8`, `n_GPe = 8`
- Keep all per-cell conductance parameters identical — the literature values are per-cell, not per-population
- Coupling weights stay the same (`g_AMPA`, `g_GABA`) — each cell still receives one mean synaptic input; the larger population provides more stable mean-field averaging

#### `src/simulation/model.py`
- No equation changes — the model already vectorises over `n_STN` / `n_GPe` cells using NumPy slicing
- Add sigmoid clipping to prevent `exp` overflow: replace `np.exp(x)` with `np.exp(np.clip(x, -500, 500))` in `_sigm()` and the two tau functions that hit overflow warnings
- State vector grows from 12 → 96 variables (8×6 + 8×6); runner and lfp are unaffected

#### `src/simulation/runner.py`
- No changes needed — already parameterised by `config.n_STN` / `config.n_GPe`

#### `src/simulation/lfp.py`
- No changes needed — already averages across all `n_STN` voltage columns

#### `tests/test_simulation.py`
- Update `test_y_shape` expected `n_state` to reflect 8+8 default
- Update `test_lfp_indices_correct` fixture if it uses `n_STN=3` (standalone, unaffected by default)
- Chunked parity tolerance: keep at 1.0 mV — population averaging makes drift smaller, not larger

#### `notebooks/01_sim_validation.ipynb`
- No structural changes; re-run after code fixes to verify sanity gate passes

---

## 1. `configs/sim_config.py`

Dataclass or plain dict with:
- **STN parameters**: `C=1.0`, `k_Na=1.0`, conductances `g_Na`, `g_K`, `g_L`, `g_Ca`, `g_T`, `g_AHP`, `E_Na`, `E_K`, `E_L`, `E_Ca`
- **GPe parameters**: same set, GPe-tuned values
- **Coupling parameters**:
  - `g_AMPA` (STN→GPe excitatory synaptic weight)
  - `g_GABA` (GPe→STN inhibitory synaptic weight)
  - `n_STN`, `n_GPe` — population sizes (MVP: 1 STN cell + 1 GPe cell)
- **Regime switch**:
  - `healthy`: nominal coupling (weak GPe→STN inhibition)
  - `pathological`: elevated `g_GABA` / modified coupling → produces beta synchrony
- **Solver settings**: `t_span`, `dt_max`, `method='RK45'`
- **Seed**: integer, controls `np.random.default_rng` for initial conditions

Use Terman-Rubin 2002 parameter values as the default (well-documented in literature, produces ~20 Hz beta in pathological regime).

---

## 2. `src/simulation/model.py`

One function: `stn_gpe_rhs(t, y, params, stim_fn)` — the ODE right-hand side passed to `solve_ivp`.

**State vector `y`**: interleaved per-neuron state variables.
Each STN cell: `[V_STN, h, n, r, Ca, s_AMPA]`
Each GPe cell: `[V_GPe, h, n, r, s_GABA]`

**Equations** (Terman-Rubin 2002):
- STN: `C dV/dt = -I_L - I_K - I_Na - I_T - I_Ca - I_AHP - I_GABA_from_GPe + I_stim`
- GPe: `C dV/dt = -I_L - I_K - I_Na - I_T - I_Ca - I_AHP - I_AMPA_from_STN`
- Gating variable ODEs: standard `dx/dt = phi*(x_inf(V) - x) / tau_x(V)` form
- Synaptic: `ds/dt = alpha * (1 - s) * H_inf(V_pre) - beta * s`

`stim_fn(t)` — callable that returns scalar stimulation current at time `t`. Default: `lambda t: 0.0`. This is the injection point for Phase 3/4 (open-loop and closed-loop stim).

Keep all math explicit and vectorized over `n_STN` / `n_GPe` cells using NumPy slicing.

---

## 3. `src/simulation/lfp.py`

Single function: `extract_lfp(y_chunk, params) -> np.ndarray`

- Takes a state matrix `y_chunk` (shape: `[n_timesteps, n_state_vars]`)
- Returns a 1-D array: **population-mean STN membrane voltage**
  - `lfp = y_chunk[:, stn_V_indices].mean(axis=1)`
- This is the one scalar LFP surrogate per the experiment spec

No filtering here — filtering is the controller's responsibility (causal only, locked in experiment contract).

---

## 4. `src/simulation/runner.py`

Two public functions:

### `run_trajectory(config, regime, seed, stim_fn=None) -> dict`
- Sets initial conditions by sampling near rest from `np.random.default_rng(seed)`
- Calls `solve_ivp` with `method='RK45'`, `max_step=config.dt_max`
- Produces full trajectory at variable time steps
- Extracts LFP via `lfp.extract_lfp`
- Resamples to 250 Hz via `scipy.interpolate.interp1d` (kind='linear') on `t_out` → uniform grid
- Returns: `{'t': ..., 'lfp': ..., 'y_raw': ..., 't_raw': ..., 'regime': regime, 'seed': seed}`

### `run_chunked(config, regime, seed, chunk_duration_s, stim_fn=None) -> dict`
- Splits simulation into `chunk_duration_s`-length chunks (needed for closed-loop control injection)
- Carries state across chunks: end state of chunk N = initial condition of chunk N+1
- Resamples each chunk independently then concatenates (boundary: discard 1 overlap sample)
- Returns same dict structure as `run_trajectory`

### `check_signal_safety(t, lfp) -> None`
Called internally after resampling. Raises `ValueError` if:
- any NaN or Inf in `lfp`
- any non-monotonic timestamps in `t`
- any discontinuity at chunk boundaries (|jump| > 5 std of signal)

---

## 5. `notebooks/01_sim_validation.ipynb`

Verification notebook — not production code. Cells:
1. Run healthy trajectory → plot LFP trace
2. Run pathological trajectory → plot LFP trace
3. PSD comparison plot (both regimes, mark 13–30 Hz beta band)
4. Confirm: pathological beta power > healthy beta power (print ratio)
5. Verify resampled stream is at exactly 250 Hz
6. Confirm no chunk-boundary artifacts (plot chunk boundaries on trace)

---

## 6. `tests/test_simulation.py`

Unit tests for structural/correctness invariants — things that must hold
regardless of parameter tuning. **Not** a regime-separation test (that
lives in the notebook and depends on parameter values).

Test cases:
- **Deterministic seeding**: `run_trajectory(config, seed=42)` called twice
  returns bit-identical `lfp` arrays
- **Output shape**: `lfp` length = `(t_end - t_warmup) / 1000 * fs` ± 1
- **Sample rate**: `np.diff(t)` is uniform at 4 ms (± floating point)
- **run_trajectory vs run_chunked parity**: both produce matching `lfp`
  arrays (within float tolerance) for the same config/seed
- **Safety check — NaN**: `check_signal_safety` raises `ValueError` on an
  array containing NaN
- **Safety check — non-monotonic t**: raises `ValueError` on reversed timestamps
- **Safety check — discontinuity**: raises `ValueError` on a signal with an
  artificial spike jump > 5 std
- **healthy_config / pathological_config**: both complete without error and
  return finite output across 3 seeds

---

## Verification (Phase 3 Sanity Gate Criteria)

Per `docs/planning/backlog.md` Phase 3, the simulator passes when:
- [x] Healthy and pathological runs are stable (no divergence/NaN) — passing with 1+1
- [ ] Pathological beta power > healthy beta power (measure 13–30 Hz band via Welch PSD) — **FAILED with 1+1 (0.67×); target after 8+8 upgrade**
- [x] Stream is fixed-rate at exactly 250 Hz after resampling — passing
- [x] Windows align with 50 ms decision cadence (128-sample windows, 12-sample stride) — passing
- [x] No visible chunk-boundary artifacts — passing
- [ ] Suppression behavior is reproducible across ≥3 seeds — pending beta fix
