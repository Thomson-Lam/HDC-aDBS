"""

LFP surrogate extraction from ODE state.

What is an LFP?
---------------
In real experiments, a Local Field Potential (LFP) is recorded by an
electrode placed in brain tissue.  It picks up the summed electrical
activity of all neurons nearby — mostly their synaptic currents and
subthreshold membrane fluctuations.

In our simulation we use a simple, widely-accepted surrogate:
  LFP(t) = mean of all STN membrane voltages at time t

This is a population-average voltage.  It is:
  - A scalar per time step (one number per window position)
  - In mV
  - Dominated by beta-frequency content when the network is pathological

Why only STN voltages?
----------------------
STN is the population we are "listening to" with the simulated electrode.
GPe is considered an upstream driver.  The experiment spec locks this
choice for the entire project — do not add GPe to the LFP without
updating the spec.

No filtering here
-----------------
The experiment contract requires causal filtering only, and filtering
is the controller's responsibility.  This module just extracts the raw
signal.  Filtering in a pre-processing step would violate causality
because it would use future samples to compute past values.
"""

import numpy as np

from configs.sim_config import SimConfig, STN_STATE_DIM


def stn_voltage_indices(config: SimConfig) -> list[int]:
    """
    Return the flat-array indices of all STN membrane voltages.

    In the state vector, each STN cell's voltage is at position
    i * STN_STATE_DIM + 0 (voltage is always the first variable per cell).

    Example
    -------
    n_STN=1 → [0]
    n_STN=3 → [0, 6, 12]
    """
    # Voltage (V) is variable index 0 within each cell's block
    return [i * STN_STATE_DIM for i in range(config.n_STN)]


def extract_lfp(y: np.ndarray, config: SimConfig) -> np.ndarray:
    """
    Extract the LFP surrogate from a state matrix.

    Parameters
    ----------
    y      : ndarray, shape (n_timesteps, n_state_vars)
             The resampled state matrix output by the runner.
             Rows are time steps; columns are state variables in the
             same interleaved layout as the ODE state vector.
    config : SimConfig

    Returns
    -------
    lfp : ndarray, shape (n_timesteps,)
          Population-mean STN membrane voltage in mV.
          One scalar per time step — this is the signal seen by
          both the classical and HDC controllers.
    """
    v_indices = stn_voltage_indices(config)   # which columns hold STN voltages

    # y[:, v_indices] → shape (n_timesteps, n_STN)
    # .mean(axis=1)   → shape (n_timesteps,)
    return y[:, v_indices].mean(axis=1)
