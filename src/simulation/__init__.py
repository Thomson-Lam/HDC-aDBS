"""Simulation package: ODE model, LFP extraction, and trajectory runner."""

from .runner import run_trajectory, run_chunked, check_signal_safety
from .lfp import extract_lfp
from .model import stn_gpe_rhs

__all__ = [
    "run_trajectory",
    "run_chunked",
    "check_signal_safety",
    "extract_lfp",
    "stn_gpe_rhs",
]
