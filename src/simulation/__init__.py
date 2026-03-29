"""
Core simulation package: ODE model, LFP extraction, and trajectory runner.
"""

from .runner import run_trajectory, run_chunked, check_signal_safety
from .lfp import extract_lfp
from .model import stn_gpe_rhs
from .open_loop_sanity import (
    OpenLoopGateConfig,
    beta_power_welch,
    make_pulse_train_stim,
    run_open_loop_sanity_gate,
)

__all__ = [
    "run_trajectory",
    "run_chunked",
    "check_signal_safety",
    "extract_lfp",
    "stn_gpe_rhs",
    "OpenLoopGateConfig",
    "make_pulse_train_stim",
    "beta_power_welch",
    "run_open_loop_sanity_gate",
]
