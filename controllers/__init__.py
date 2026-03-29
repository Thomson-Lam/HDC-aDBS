"""
controllers — closed-loop DBS controller library.

Public API
----------
HDCController    : closed-loop controller using a trained HDC classifier
BetaController   : classical baseline using causal beta-band RMS thresholding
run_closed_loop  : simulation harness that wires a controller to the ODE runner
ControllerConfig : frozen dataclass with all timing / threshold parameters
StimState        : enum — IDLE / STIMULATING / LOCKOUT
"""

from .base import ControllerConfig, ControllerMetrics, StimState
from .hdc_controller import HDCController
from .beta_controller import BetaController
from .run_controller import run_closed_loop

__all__ = [
    "ControllerConfig",
    "ControllerMetrics",
    "StimState",
    "HDCController",
    "BetaController",
    "run_closed_loop",
]
