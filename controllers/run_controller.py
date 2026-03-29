"""
Closed-loop simulation harness.

run_closed_loop() mirrors the inner loop of src/simulation/runner.run_chunked()
but inserts controller feedback between ODE chunks:

    for each 50 ms chunk:
        1. Ask the controller for the current stim_fn (closure over its state)
        2. Solve the ODE for this chunk with that stim_fn
        3. Resample the variable-step output to the global 250 Hz grid
        4. Extract the LFP surrogate (mean STN voltage)
        5. Feed the chunk to the controller → updates buffer, decisions, state
        6. Record the stim amplitude and StimState for every sample

Why replicate the chunk loop rather than calling run_chunked()?
---------------------------------------------------------------
run_chunked() accepts a single stim_fn at the start and runs the whole
simulation before returning.  For closed-loop control we need to update
stim_fn BETWEEN chunks.  Replicating the loop here gives us that control
point while keeping the same grid-alignment and state-carry logic.

The grid-alignment math (n_first / n_last) is copied verbatim from
runner.py:382-384 to guarantee that consecutive chunks tile the global
250 Hz grid without gaps or duplicates.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from configs.sim_config import SimConfig, STN_STATE_DIM, GPE_STATE_DIM
from src.simulation.model import stn_gpe_rhs
from src.simulation.lfp import extract_lfp
from src.simulation.runner import (
    resample_to_fs,
    check_signal_safety,
    _initial_conditions,  # not exported publicly; imported directly
)
from .base import BaseController

if TYPE_CHECKING:
    pass


def run_closed_loop(
    controller: BaseController,
    config: SimConfig,
    seed: int,
    chunk_duration_s: float = 0.05,
) -> dict:
    """Run the STN-GPe ODE with closed-loop controller feedback.

    The simulation is divided into chunks of chunk_duration_s.  After each
    chunk the controller ingests the new LFP samples, updates its internal
    state (buffer, state machine, metrics), and provides a fresh stim_fn for
    the next chunk.

    Parameters
    ----------
    controller       : any BaseController subclass (HDCController, BetaController)
    config           : SimConfig — ODE parameters, regime, and timing
    seed             : int — RNG seed for reproducible initial conditions
    chunk_duration_s : float seconds — chunk length (default 0.05 s = 50 ms).
                       Should equal decision_cadence_ms / 1000 so the controller
                       makes exactly one decision per chunk.

    Returns
    -------
    dict with keys:
        't'       : (n_samples,) ms — uniform 250 Hz time axis, warmup discarded
        'lfp'     : (n_samples,) mV — raw LFP (mean STN voltage), warmup discarded
        'stim'    : (n_samples,) float — stim current active at each sample (μA/cm²)
        'state'   : (n_samples,) int — StimState.value at each sample (0/1/2)
        'metrics' : ControllerMetrics — accumulated statistics (call .to_dict())
        'config'  : SimConfig used
        'seed'    : int seed used
    """
    dt_ms = 1000.0 / config.fs  # 4.0 ms at 250 Hz
    chunk_ms = chunk_duration_s * 1000.0

    # Seeded initial conditions (same as runner.py so results are comparable)
    rng = np.random.default_rng(seed)
    y_current = _initial_conditions(config, rng)

    # Build chunk time boundaries: [t0, t1), [t1, t2), ..., [t_{n-1}, t_end)
    t_starts = np.arange(config.t_start, config.t_end, chunk_ms)
    t_ends = np.append(t_starts[1:], config.t_end)

    # Per-chunk accumulators (we'll concatenate at the end)
    t_parts = []  # (n_chunk_samples,) ms
    lfp_parts = []  # (n_chunk_samples,) mV  — raw LFP, warmup not yet stripped
    stim_parts = []  # (n_chunk_samples,) float — stim current for each sample
    state_parts = []  # (n_chunk_samples,) int  — StimState.value for each sample

    for t_a, t_b in zip(t_starts, t_ends):
        if t_a >= t_b:
            continue  # skip zero-length edge case at the very end of the run

        # ---- 1. Get current stim_fn from controller ----------------------
        # The closure captures controller.state by reference so the ODE solver
        # sees the state that was current at the START of this chunk.  The
        # state does not change mid-chunk (controller.ingest() is called after
        # the ODE completes), which is the correct causal behaviour.
        stim_fn = controller.get_stim_fn()

        # ---- 2. Solve ODE for this chunk ---------------------------------
        def rhs(t, y):
            return stn_gpe_rhs(t, y, config, stim_fn)

        sol = solve_ivp(
            rhs,
            [t_a, t_b],
            y_current,
            method="RK45",
            max_step=config.dt_max,  # must resolve action potentials
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"ODE solver failed in chunk [{t_a:.1f}, {t_b:.1f}] ms: {sol.message}"
            )

        # ---- 3. Resample to global 250 Hz grid ---------------------------
        # Use the same grid-alignment math as runner.py:382-384.
        # n_first / n_last define which global-grid indices fall within [t_a, t_b).
        # Consecutive chunks therefore tile the grid with no gaps or duplicates.
        n_first = int(math.ceil((t_a - config.t_start) / dt_ms))
        n_last = int(math.ceil((t_b - config.t_start) / dt_ms)) - 1
        t_chunk_grid = config.t_start + np.arange(n_first, n_last + 1) * dt_ms

        t_c, y_c = resample_to_fs(sol.t, sol.y, config.fs, t_grid=t_chunk_grid)

        # ---- 4. Extract LFP surrogate ------------------------------------
        lfp_c = extract_lfp(y_c, config)

        # Record stim/state that the ODE actually saw in this chunk.
        # This must happen before ingest() mutates controller state.
        stim_chunk = np.asarray([stim_fn(float(t_i)) for t_i in t_c], dtype=np.float64)
        state_chunk = np.asarray(
            [controller.state_at(float(t_i)).value for t_i in t_c], dtype=np.int32
        )

        # ---- 5. Feed chunk to controller ---------------------------------
        # Only feed samples that are past the warmup window.  During warmup
        # the ODE is settling from initial conditions; we don't want those
        # transients filling the controller's ring buffer.
        if t_b > config.t_warmup:
            controller.ingest(t_c, lfp_c)

        # ---- 6. Accumulate output arrays ---------------------------------
        n_c = len(t_c)
        t_parts.append(t_c)
        lfp_parts.append(lfp_c)
        stim_parts.append(stim_chunk)
        state_parts.append(state_chunk)

        # ---- 7. Carry raw ODE end-state (not resampled) ------------------
        # Using the last raw ODE point avoids accumulating interpolation error
        # across chunk boundaries, matching the runner.py convention.
        y_current = sol.y[:, -1]

    # Concatenate all chunks
    t_full = np.concatenate(t_parts)
    lfp_full = np.concatenate(lfp_parts)
    stim_full = np.concatenate(stim_parts)
    state_full = np.concatenate(state_parts)

    # Discard warmup transient (same logic as run_trajectory / run_chunked)
    n_warmup = int(math.ceil(config.t_warmup / dt_ms))
    t_out = t_full[n_warmup:]
    lfp_out = lfp_full[n_warmup:]
    stim_out = stim_full[n_warmup:]
    state_out = state_full[n_warmup:]

    # Safety check on the raw LFP (matches runner.py convention)
    check_signal_safety(t_out, lfp_out)

    return {
        "t": t_out,
        "lfp": lfp_out,
        "stim": stim_out,
        "state": state_out,
        "metrics": controller.metrics,
        "config": config,
        "seed": seed,
    }
