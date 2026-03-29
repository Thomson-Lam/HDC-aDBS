"""Simulation runner: solve the ODE, resample to 250 Hz, validate output.

Two entry points
----------------
run_trajectory(config, seed)
    Runs the full simulation in one shot.  Simple and fast.
    Use this for offline data generation (Phase 2 onwards).

run_chunked(config, seed, chunk_duration_s)
    Runs the simulation in fixed-length chunks, carrying the ODE state
    across chunk boundaries.  This is the building block for closed-loop
    control (Phase 4): between chunks, a controller can inspect the
    signal and update the stim_fn.

Both functions
    1. Set initial conditions near physiological rest (seeded RNG)
    2. Call scipy solve_ivp (RK45, variable step)
    3. Resample the raw variable-step output to a uniform 250 Hz grid
    4. Extract the LFP surrogate (mean STN voltage)
    5. Discard the warmup transient
    6. Run safety checks on the output stream

Return value
------------
Both functions return a dict:
  't'      : (n_samples,) ms — uniform time grid, warmup discarded
  'lfp'    : (n_samples,) mV — LFP signal ready for windowing
  'y'      : (n_samples, n_state) — full state, warmup discarded
  'y_raw'  : (n_raw_total, n_state) — raw ODE output (variable step)
  't_raw'  : (n_raw_total,) ms     — raw ODE time points
  'config' : SimConfig used
  'seed'   : int seed used
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from configs.sim_config import SimConfig, STN_STATE_DIM, GPE_STATE_DIM
from .model import stn_gpe_rhs
from .lfp import extract_lfp


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

# Approximate physiological rest values for each cell type.
# The ODE will quickly settle into its limit cycle regardless of where we
# start, but starting near rest reduces the required warmup time.
_STN_REST = np.array([-60.0,  # V   (mV)  — near resting potential
                        0.4,  # h        — partially inactivated Na
                        0.1,  # n        — mostly closed K channels
                       0.25,  # r        — partially available T-Ca
                       0.05,  # Ca  (mM) — low baseline Ca
                        0.0]) # s_AMPA   — no ongoing release at rest

_GPE_REST = np.array([-65.0,  # V
                        0.4,  # h
                        0.1,  # n
                        0.5,  # r   — GPe has more T-Ca available at rest
                       0.05,  # Ca
                        0.0]) # s_GABA


def _initial_conditions(config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Build the initial state vector y0 by sampling near rest values.

    A small amount of Gaussian noise is added to each cell's voltage to
    break symmetry when n_STN > 1 or n_GPe > 1.  Without noise, identical
    cells in the same population would stay perfectly synchronised forever
    regardless of coupling strength — an unphysical artefact.

    Parameters
    ----------
    config : SimConfig
    rng    : seeded Generator for reproducibility

    Returns
    -------
    y0 : (n_STN*6 + n_GPe*6,) initial state vector
    """
    n_s = config.n_STN
    n_g = config.n_GPe
    y0 = np.zeros(n_s * STN_STATE_DIM + n_g * GPE_STATE_DIM)

    for i in range(n_s):
        noise = rng.normal(0.0, 0.1, STN_STATE_DIM)
        noise[0] *= 5.0   # voltage noise is larger (±0.5 mV std) so cells desynchronise faster
        start = i * STN_STATE_DIM
        y0[start : start + STN_STATE_DIM] = _STN_REST + noise

    gpe_offset = n_s * STN_STATE_DIM
    for j in range(n_g):
        noise = rng.normal(0.0, 0.1, GPE_STATE_DIM)
        noise[0] *= 5.0
        start = gpe_offset + j * GPE_STATE_DIM
        y0[start : start + GPE_STATE_DIM] = _GPE_REST + noise

    return y0


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_fs(t_raw: np.ndarray, y_raw: np.ndarray, fs: float,
                   t_grid: np.ndarray | None = None):
    """
    Interpolate variable-step ODE output onto a uniform time grid at fs Hz.

    The ODE solver (RK45) adapts its step size automatically — small steps
    during fast events (spikes), large steps during slow periods (inter-spike
    intervals).  The downstream signal pipeline and controllers need a
    fixed-rate stream, so we interpolate.

    We use linear interpolation.  Action potentials are smooth on the
    timescale of the ODE steps (solver ensures local error ≤ tolerance),
    so linear interpolation between the dense solver output is accurate
    enough and fast.

    Parameters
    ----------
    t_raw  : (n_raw,) ms      — variable-step time points from solve_ivp
    y_raw  : (n_state, n_raw) — state matrix in scipy convention (rows=vars)
    fs     : float Hz         — target sample rate
    t_grid : (n_samples,) ms, optional
             Pre-computed uniform grid to interpolate onto.
             If None, a local grid starting at t_raw[0] is created.
             Pass this from run_chunked so all chunks share the same global
             250 Hz grid anchored at config.t_start — prevents inter-chunk
             gaps from misaligned arange() endpoints.

    Returns
    -------
    t_grid    : (n_samples,) ms      — equally spaced time points
    y_uniform : (n_samples, n_state) — state at each uniform time point
    """
    if t_grid is None:
        dt_ms  = 1000.0 / fs     # e.g. 4.0 ms at 250 Hz
        t_grid = np.arange(t_raw[0], t_raw[-1], dt_ms)

    # interp1d axis=1: each row is one state variable; interpolate along columns (time)
    f = interp1d(t_raw, y_raw, kind='linear', axis=1, assume_sorted=True)
    y_uniform = f(t_grid)        # shape (n_state, n_samples)

    return t_grid, y_uniform.T   # transpose to (n_samples, n_state)


# Backward-compatible alias — internal code that still calls _resample_to_fs will work.
_resample_to_fs = resample_to_fs


# ---------------------------------------------------------------------------
# Signal safety checks  (Backlog Phase 2, unit 5)
# ---------------------------------------------------------------------------

def check_signal_safety(t: np.ndarray, lfp: np.ndarray) -> None:
    """
    Validate the resampled LFP stream before it leaves the runner.

    Raises ValueError immediately if any problem is found so the caller
    knows the trajectory is unusable rather than silently propagating bad
    data into the dataset or training pipeline.

    Checks
    ------
    1. NaN / Inf — indicates ODE divergence (step size collapsed, voltage
       blew up).  Common cause: parameter misconfiguration or too-large dt_max.

    2. Non-monotonic timestamps — would cause interp1d to produce garbage.
       Should never happen from solve_ivp output, but caught here defensively.

    3. Large discontinuity — a sudden jump larger than 5× the signal std
       suggests a chunk boundary artefact (if using run_chunked) or a
       pathological initial condition.  Threshold of 5 std is loose enough
       to allow normal spike upstrokes but tight enough to catch artefacts.

    Parameters
    ----------
    t   : (n_samples,) ms   — uniform time grid
    lfp : (n_samples,) mV   — LFP signal to validate
    """
    # Check 1: finite values
    if not np.all(np.isfinite(lfp)):
        n_bad = np.sum(~np.isfinite(lfp))
        raise ValueError(
            f"LFP contains {n_bad} non-finite value(s) (NaN or Inf). "
            "The ODE likely diverged — check dt_max and parameter values."
        )

    # Check 2: monotonic timestamps
    if not np.all(np.diff(t) > 0):
        raise ValueError(
            "Timestamps are not strictly monotonic. "
            "This should not happen with solve_ivp output."
        )

    # Check 3: discontinuities (only meaningful with > 1 sample)
    #
    # Threshold = max(5 × signal std,  150 mV hard floor).
    #
    # Why the floor?  At 250 Hz (4 ms per sample) a genuine action potential
    # upstroke can be up to ~115 mV (resting ≈ -60 mV → peak ≈ +55 mV).
    # Without the floor, a short signal with low std (e.g. only 25 samples)
    # can produce a threshold below that, causing real spikes to be flagged
    # as artefacts.  150 mV is above any physiologically possible single-step
    # transition in this model, so only genuine ODE blow-ups or badly carried
    # chunk boundaries will trigger it.
    if len(lfp) > 1:
        jumps     = np.abs(np.diff(lfp))
        threshold = max(5.0 * lfp.std(), 150.0)
        bad_idx   = np.where(jumps > threshold)[0]
        if len(bad_idx) > 0:
            raise ValueError(
                f"Signal discontinuity detected at {len(bad_idx)} location(s). "
                f"Largest jump: {jumps[bad_idx].max():.2f} mV  "
                f"(threshold: {threshold:.2f} mV = max(5 × std, 150 mV)). "
                "If using run_chunked, check that chunk-boundary state is "
                "carried correctly."
            )


# ---------------------------------------------------------------------------
# Public API: single trajectory
# ---------------------------------------------------------------------------

def run_trajectory(
    config: SimConfig,
    seed: int,
    stim_fn=None,
) -> dict:
    """
    Run the full simulation as one solve_ivp call.

    This is the simplest way to generate a trajectory.  The ODE is solved
    from t_start to t_end in one go; there is no opportunity to inject
    control decisions mid-run.  Use run_chunked for that.

    Parameters
    ----------
    config  : SimConfig — includes regime (via g_GABA) and timing
    seed    : int       — RNG seed; same seed → same initial conditions → same trajectory
    stim_fn : callable(t) -> float, optional
              Stimulation current (μA/cm²) as a function of time.
              For open-loop testing (Phase 3) this could be a square wave.
              For closed-loop (Phase 4) use run_chunked instead.

    Returns
    -------
    dict — see module docstring for key descriptions
    """
    rng = np.random.default_rng(seed)
    y0  = _initial_conditions(config, rng)

    # Wrap the RHS so solve_ivp only sees (t, y)
    def rhs(t, y):
        return stn_gpe_rhs(t, y, config, stim_fn)

    sol = solve_ivp(
        rhs,
        [config.t_start, config.t_end],
        y0,
        method='RK45',
        max_step=config.dt_max,   # caps step size to ensure spikes are resolved
        dense_output=False,       # we do our own resampling via interp1d
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # sol.t : (n_raw,)           — variable-step time points
    # sol.y : (n_state, n_raw)   — state at each time point (scipy convention)

    t_uniform, y_uniform = resample_to_fs(sol.t, sol.y, config.fs)
    lfp_full = extract_lfp(y_uniform, config)

    # Discard warmup transient
    # dt_ms * n_warmup ≈ config.t_warmup  (round up to avoid using any warmup samples)
    dt_ms    = 1000.0 / config.fs
    n_warmup = int(np.ceil(config.t_warmup / dt_ms))
    t_out    = t_uniform[n_warmup:]
    lfp_out  = lfp_full[n_warmup:]
    y_out    = y_uniform[n_warmup:]

    check_signal_safety(t_out, lfp_out)

    return {
        't':      t_out,
        'lfp':    lfp_out,
        'y':      y_out,
        'y_raw':  sol.y.T,        # (n_raw, n_state) for inspection
        't_raw':  sol.t,
        'config': config,
        'seed':   seed,
    }


# ---------------------------------------------------------------------------
# Public API: chunked trajectory  (closed-loop ready)
# ---------------------------------------------------------------------------

def run_chunked(
    config: SimConfig,
    seed: int,
    chunk_duration_s: float = 0.5,
    stim_fn=None,
) -> dict:
    """
    Run the simulation in fixed-length chunks, carrying state across boundaries.

    Why chunked?
    ------------
    A closed-loop controller needs to see the latest signal window, make a
    decision, and then update the stimulation before the next chunk runs.
    run_trajectory cannot do this because the entire trajectory is solved at
    once.  run_chunked solves one chunk at a time; between chunks, the caller
    can inspect the LFP and mutate stim_fn (or swap it out).

    In Phase 2 (offline data generation) this produces the same output as
    run_trajectory — the stim_fn is None and state is carried continuously.
    The chunk boundary is invisible in the resampled output (one overlap
    sample is dropped per boundary to avoid duplicates).

    Parameters
    ----------
    config           : SimConfig
    seed             : int
    chunk_duration_s : float seconds — length of each chunk
                       0.5 s = 500 ms → 125 samples at 250 Hz per chunk
                       Must be long enough to contain at least a few spikes.
    stim_fn          : callable(t) -> float, optional

    Returns
    -------
    Same dict structure as run_trajectory.
    """
    rng      = np.random.default_rng(seed)
    y_current = _initial_conditions(config, rng)
    dt_ms    = 1000.0 / config.fs

    # Build chunk time boundaries  [t0, t1), [t1, t2), ..., [t_{n-1}, t_end)
    chunk_ms = chunk_duration_s * 1000.0
    t_starts = np.arange(config.t_start, config.t_end, chunk_ms)
    t_ends   = np.append(t_starts[1:], config.t_end)

    # Accumulators
    t_parts   = []
    y_parts   = []
    t_raw_all = []
    y_raw_all = []

    for t_a, t_b in zip(t_starts, t_ends):
        if t_a >= t_b:
            continue   # skip zero-length edge case at the very end

        def rhs(t, y):
            return stn_gpe_rhs(t, y, config, stim_fn)

        sol = solve_ivp(
            rhs,
            [t_a, t_b],
            y_current,
            method='RK45',
            max_step=config.dt_max,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"ODE solver failed in chunk [{t_a:.1f}, {t_b:.1f}] ms: {sol.message}"
            )

        # Build a globally-aligned grid for this chunk.
        #
        # Problem with naive np.arange(t_a, t_b, dt_ms):
        #   chunk [0, 50] ms ends at 48 ms (arange stops before 50)
        #   chunk [50,100] ms starts at 50, drop first → first kept = 54 ms
        #   gap 48 → 54 = 6 ms  ≠  4 ms
        #
        # Fix: compute which global grid indices fall in [t_a, t_b):
        #   n_first = ceil((t_a - t_start) / dt_ms)  → first index ≥ t_a
        #   n_last  = ceil((t_b - t_start) / dt_ms) - 1  → last index < t_b
        # This way consecutive chunks tile the global grid with no gaps or overlaps.
        n_first      = int(np.ceil((t_a - config.t_start) / dt_ms))
        n_last       = int(np.ceil((t_b - config.t_start) / dt_ms)) - 1
        t_chunk_grid = config.t_start + np.arange(n_first, n_last + 1) * dt_ms

        t_c, y_c = resample_to_fs(sol.t, sol.y, config.fs, t_grid=t_chunk_grid)

        t_parts.append(t_c)
        y_parts.append(y_c)
        t_raw_all.append(sol.t)
        y_raw_all.append(sol.y.T)

        # Carry the last raw ODE state (not the interpolated one) into
        # the next chunk.  Using the interpolated state would accumulate
        # error from repeated resampling.
        y_current = sol.y[:, -1]

    t_uniform = np.concatenate(t_parts)
    y_uniform = np.vstack(y_parts)
    lfp_full  = extract_lfp(y_uniform, config)

    # Discard warmup
    n_warmup = int(np.ceil(config.t_warmup / dt_ms))
    t_out    = t_uniform[n_warmup:]
    lfp_out  = lfp_full[n_warmup:]
    y_out    = y_uniform[n_warmup:]

    check_signal_safety(t_out, lfp_out)

    return {
        't':      t_out,
        'lfp':    lfp_out,
        'y':      y_out,
        'y_raw':  np.vstack(y_raw_all),
        't_raw':  np.concatenate(t_raw_all),
        'config': config,
        'seed':   seed,
    }
