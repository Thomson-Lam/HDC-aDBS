"""Shared stimulation waveform helpers for closed-loop controllers."""

from __future__ import annotations

import math
from collections.abc import Callable


def pulse_train_value(
    t_ms: float,
    amplitude: float,
    frequency_hz: float,
    pulse_width_ms: float,
) -> float:
    """Return instantaneous pulse-train amplitude at time ``t_ms``.

    Pulse train convention:
    - period is ``1000 / frequency_hz`` milliseconds
    - output is ``amplitude`` for ``phase < pulse_width_ms``
    - output is ``0`` otherwise
    """
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0")
    if pulse_width_ms <= 0.0:
        raise ValueError("pulse_width_ms must be > 0")
    period_ms = 1000.0 / frequency_hz
    phase = math.fmod(t_ms, period_ms)
    if phase < 0.0:
        phase += period_ms
    return amplitude if phase < pulse_width_ms else 0.0


def make_epoch_gated_pulse_train(
    *,
    amplitude: float,
    frequency_hz: float,
    pulse_width_ms: float,
    epoch_active_fn: Callable[[float], bool],
) -> Callable[[float], float]:
    """Create ``stim_fn(t_ms)`` that gates a pulse train by epoch activity."""

    def stim_fn(t_ms: float) -> float:
        if not epoch_active_fn(t_ms):
            return 0.0
        return pulse_train_value(
            t_ms=t_ms,
            amplitude=amplitude,
            frequency_hz=frequency_hz,
            pulse_width_ms=pulse_width_ms,
        )

    return stim_fn


def pulses_per_epoch(stim_duration_ms: float, frequency_hz: float) -> int:
    """Return expected pulse starts per epoch.

    The project's frozen defaults (200 ms, 130 Hz) yield exactly 26 pulses.
    """
    if stim_duration_ms <= 0.0:
        return 0
    return int(round(stim_duration_ms * frequency_hz / 1000.0))
