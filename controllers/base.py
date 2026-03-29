"""
Base classes and shared data structures for all closed-loop DBS controllers.

Every controller (HDC, beta-power, future variants) shares the same mechanics:
  - A rolling ring buffer that holds the last window_length LFP samples
  - A three-state stimulation machine:  IDLE → STIMULATING → LOCKOUT → IDLE
  - Decision scoring at a fixed cadence (every decision_cadence_ms)
  - A stim_fn closure used by the ODE solver to inject current into STN

Subclasses only need to implement _compute_decision(window) → float.
All buffering, state transitions, and metric accumulation live here.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from .waveform import make_epoch_gated_pulse_train, pulses_per_epoch


# ---------------------------------------------------------------------------
# Stimulation state machine states
# ---------------------------------------------------------------------------


class StimState(Enum):
    """Three-state machine controlling when stimulation current is applied.

    IDLE        — no pathology detected; no current injected
    STIMULATING — a burst is active; stim_amplitude is injected until _stim_end_t
    LOCKOUT     — post-burst refractory period; no new trigger allowed until
                  _lockout_end_t so we don't chatter
    """

    IDLE = 0
    STIMULATING = 1
    LOCKOUT = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerConfig:
    """All tunable parameters for the control loop.

    Defaults match the locked experiment spec (docs/experiment-spec.md).
    Override at instantiation time; the dataclass is frozen so it cannot
    be mutated after construction.
    """

    fs: float = 250.0
    """Sampling rate (Hz).  Must match SimConfig.fs."""

    window_length: int = 128
    """Decision window size in samples (512 ms at 250 Hz)."""

    decision_cadence_ms: float = 50.0
    """How often the controller evaluates the latest window (ms).
    At 250 Hz this is 12–13 samples between decisions."""

    stim_duration_ms: float = 200.0
    """Length of one stimulation burst (ms)."""

    lockout_duration_ms: float = 200.0
    """Post-burst refractory period before the next trigger is allowed (ms).
    Total minimum inter-trigger gap = stim_duration_ms + lockout_duration_ms."""

    stim_amplitude: float = 1.5
    """Stimulation current injected into all STN cells (μA/cm²).
    Positive values are excitatory in the STN model convention."""

    threshold: float = 0.0
    """Decision score threshold.  score >= threshold triggers stim.
    For HDC margin scores the natural zero-crossing separates healthy (negative)
    from pathological (positive); 0.0 is the sensible default before calibration."""

    pulse_frequency_hz: float = 130.0
    """Pulse frequency used inside each stimulation epoch (Hz)."""

    pulse_width_ms: float = 1.0
    """Pulse width used inside each stimulation epoch (ms)."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class ControllerMetrics:
    """Running totals accumulated during a closed-loop simulation.

    All counts and timestamps are in units of the simulation time axis (ms).
    """

    n_decisions: int = 0
    """Total number of times _compute_decision() was called."""

    n_detections: int = 0
    """Times the decision score crossed the threshold (regardless of state)."""

    n_stimulations: int = 0
    """Actual stim bursts triggered (detections that were NOT blocked by lockout)."""

    total_stim_samples: int = 0
    """Cumulative samples spent in STIMULATING state; numerator of duty cycle."""

    total_samples: int = 0
    """Total samples ingested; denominator of duty cycle."""

    stim_times_ms: list = field(default_factory=list)
    """Simulation timestamps (ms) of each triggered stim burst."""

    pulse_count: int = 0
    """Total pulse starts delivered across stimulation epochs."""

    blocked_detections_lockout: int = 0
    """Detections ignored while in LOCKOUT."""

    blocked_detections_stimulating: int = 0
    """Detections ignored while already STIMULATING."""

    decision_times_ms: list = field(default_factory=list)
    """Per-decision compute times (ms) for _compute_decision."""

    @property
    def stim_onset_times_ms(self) -> list:
        """Alias for stim_times_ms for clearer benchmark naming."""
        return self.stim_times_ms

    @property
    def duty_cycle(self) -> float:
        """Fraction of time the stimulator was active (0.0–1.0)."""
        if self.total_samples == 0:
            return 0.0
        return self.total_stim_samples / self.total_samples

    def to_dict(self) -> dict:
        """Serialise to a plain dict for YAML / JSON output."""
        decision_times = np.asarray(self.decision_times_ms, dtype=np.float64)
        decision_time_mean_ms = (
            float(decision_times.mean()) if decision_times.size else 0.0
        )
        decision_time_p95_ms = (
            float(np.percentile(decision_times, 95.0)) if decision_times.size else 0.0
        )
        return {
            "n_decisions": self.n_decisions,
            "n_detections": self.n_detections,
            "n_stimulations": self.n_stimulations,
            "duty_cycle": self.duty_cycle,
            "total_stim_samples": self.total_stim_samples,
            "total_samples": self.total_samples,
            "stim_times_ms": list(self.stim_times_ms),
            "stim_onset_times_ms": list(self.stim_times_ms),
            "pulse_count": self.pulse_count,
            "blocked_detections_lockout": self.blocked_detections_lockout,
            "blocked_detections_stimulating": self.blocked_detections_stimulating,
            "decision_times_ms": list(self.decision_times_ms),
            "decision_time_mean_ms": decision_time_mean_ms,
            "decision_time_p95_ms": decision_time_p95_ms,
        }


# ---------------------------------------------------------------------------
# Abstract base controller
# ---------------------------------------------------------------------------


class BaseController(ABC):
    """Abstract base for all closed-loop DBS controllers.

    Subclasses must implement only:
        _compute_decision(window: np.ndarray) -> float

    Everything else — ring buffer management, state machine transitions,
    metric accumulation, and the stim_fn factory — is handled here.

    Time is tracked in milliseconds throughout, consistent with SimConfig
    and the runner's time axis.
    """

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

        # ---- ring buffer -------------------------------------------------
        # Fixed-size circular buffer holding the last window_length LFP samples.
        # _buffer_head points to the slot that will be written NEXT (oldest data).
        # When the buffer is full, pushing a new sample overwrites the oldest one.
        self._buffer: np.ndarray = np.empty(config.window_length, dtype=np.float64)
        self._buffer_head: int = 0
        self._buffer_fill: int = 0  # counts 0 → window_length then stays there

        # ---- state machine -----------------------------------------------
        self.state: StimState = StimState.IDLE
        self._stim_end_t: float = -math.inf  # sim time (ms) when current burst ends
        self._lockout_end_t: float = -math.inf  # sim time (ms) when lockout ends
        self._current_t: float = 0.0  # most recent sim time seen

        # ---- decision cadence --------------------------------------------
        # Number of samples between successive _compute_decision calls.
        # round() instead of int() to stay closest to the target 50 ms cadence.
        self._decision_stride: int = max(
            1, round(config.decision_cadence_ms * config.fs / 1000.0)
        )
        self._samples_since_decision: int = 0

        # ---- metrics -----------------------------------------------------
        self.metrics: ControllerMetrics = ControllerMetrics()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest(self, t: np.ndarray, lfp: np.ndarray) -> None:
        """Feed a chunk of LFP samples into the controller.

        This is the main entry point called by the simulation harness after
        each ODE chunk completes.  For every sample it:
          1. Pushes the sample into the ring buffer.
          2. Calls _update_state_machine() to advance STIMULATING / LOCKOUT timing.
          3. Fires _compute_decision() every _decision_stride samples once the
             buffer is full.

        Parameters
        ----------
        t   : (n_samples,) ms — simulation timestamps for this chunk
        lfp : (n_samples,) mV — LFP signal for this chunk
        """
        for t_i, lfp_i in zip(t, lfp):
            t_i = float(t_i)
            self._push_sample(float(lfp_i))
            self.metrics.total_samples += 1
            self._current_t = t_i

            # Update state machine before decision so stim state is current.
            self._update_state_machine(t_i)

            self._samples_since_decision += 1

            # Evaluate once per stride, but only after the buffer is full
            # (no decision with a partially-filled window).
            if self._samples_since_decision >= self._decision_stride and self.is_ready:
                self._samples_since_decision = 0
                window = self.get_latest_window()
                t_start = time.perf_counter()
                score = self._compute_decision(window)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                self.metrics.decision_times_ms.append(float(elapsed_ms))
                self.metrics.n_decisions += 1

                if score >= self.config.threshold:
                    self.metrics.n_detections += 1
                    self._try_trigger_stim(t_i)

    def get_latest_window(self) -> np.ndarray | None:
        """Return the most recent window_length samples in chronological order.

        The ring buffer's head pointer points to the slot about to be
        overwritten, i.e. the OLDEST valid sample.  We rotate from that
        position forward to get a chronologically-ordered window.

        Returns None if fewer than window_length samples have been ingested.
        """
        if not self.is_ready:
            return None
        n = self.config.window_length
        idx = (self._buffer_head + np.arange(n)) % n
        return self._buffer[idx].copy()

    def get_stim_fn(self) -> Callable[[float], float]:
        """Return a closure that reflects the current stimulation waveform.

        This closure is passed to the ODE solver as stim_fn(t) -> float.
        It captures `self` by reference so it always reads the most recent
        StimState at the moment the ODE evaluates it.  The closure is
        rebuilt once per chunk (after controller.ingest() returns), so the
        ODE within each chunk sees a consistent state for its duration.
        """
        return make_epoch_gated_pulse_train(
            amplitude=self.config.stim_amplitude,
            frequency_hz=self.config.pulse_frequency_hz,
            pulse_width_ms=self.config.pulse_width_ms,
            epoch_active_fn=self.is_epoch_active,
        )

    def state_at(self, t_ms: float) -> StimState:
        """Predict controller state at ``t_ms`` using current state timers."""
        if self.state == StimState.IDLE:
            return StimState.IDLE
        if self.state == StimState.STIMULATING:
            if not math.isfinite(self._stim_end_t):
                return StimState.STIMULATING
            if t_ms < self._stim_end_t:
                return StimState.STIMULATING
            if t_ms < self._lockout_end_t:
                return StimState.LOCKOUT
            return StimState.IDLE
        if not math.isfinite(self._lockout_end_t):
            return StimState.LOCKOUT
        if t_ms < self._lockout_end_t:
            return StimState.LOCKOUT
        return StimState.IDLE

    def is_epoch_active(self, t_ms: float) -> bool:
        """Return whether stimulation epoch is active at ``t_ms``."""
        return self.state_at(t_ms) == StimState.STIMULATING

    @property
    def is_ready(self) -> bool:
        """True once the ring buffer contains at least window_length samples."""
        return self._buffer_fill >= self.config.window_length

    def reset(self) -> None:
        """Reset buffer, state machine, and metrics to initial state.

        Used in tests and when reusing a controller object for multiple runs.
        """
        self._buffer = np.empty(self.config.window_length, dtype=np.float64)
        self._buffer_head = 0
        self._buffer_fill = 0
        self.state = StimState.IDLE
        self._stim_end_t = -math.inf
        self._lockout_end_t = -math.inf
        self._current_t = 0.0
        self._samples_since_decision = 0
        self.metrics = ControllerMetrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_sample(self, sample: float) -> None:
        """Write one sample into the ring buffer, advancing the head pointer.

        When the buffer is full, _buffer_head wraps around and overwrites
        the oldest value — standard circular buffer write.
        """
        self._buffer[self._buffer_head] = sample
        # Advance head (wraps at window_length)
        self._buffer_head = (self._buffer_head + 1) % self.config.window_length
        # Fill counter saturates at window_length once the buffer is full
        if self._buffer_fill < self.config.window_length:
            self._buffer_fill += 1

    def _update_state_machine(self, current_t: float) -> None:
        """Advance state machine transitions based on elapsed simulation time.

        Called every sample so transitions happen at the correct wall-clock
        time (in ms) rather than on a coarser event boundary.

        Accumulates total_stim_samples while STIMULATING for duty-cycle math.
        """
        if self.state == StimState.STIMULATING:
            if current_t < self._stim_end_t:
                self.metrics.total_stim_samples += 1
            # Burst has expired → enter post-burst lockout
            if current_t >= self._stim_end_t:
                self.state = StimState.LOCKOUT

        elif self.state == StimState.LOCKOUT:
            # Lockout period expired → back to idle, ready to trigger again
            if current_t >= self._lockout_end_t:
                self.state = StimState.IDLE

    def _try_trigger_stim(self, current_t: float) -> None:
        """Attempt to start a stimulation burst at current_t (ms).

        Only succeeds from IDLE state — detections during STIMULATING or LOCKOUT
        are counted in n_detections but do NOT retrigger (avoids chatter).
        """
        if self.state != StimState.IDLE:
            if self.state == StimState.STIMULATING:
                self.metrics.blocked_detections_stimulating += 1
            elif self.state == StimState.LOCKOUT:
                self.metrics.blocked_detections_lockout += 1
            return  # blocked by ongoing stim or lockout — do nothing

        self.state = StimState.STIMULATING
        self._stim_end_t = current_t + self.config.stim_duration_ms
        self._lockout_end_t = self._stim_end_t + self.config.lockout_duration_ms
        self.metrics.n_stimulations += 1
        self.metrics.stim_times_ms.append(current_t)
        self.metrics.pulse_count += pulses_per_epoch(
            stim_duration_ms=self.config.stim_duration_ms,
            frequency_hz=self.config.pulse_frequency_hz,
        )

    # ------------------------------------------------------------------
    # Abstract: subclasses implement this only
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_decision(self, window: np.ndarray) -> float:
        """Compute a scalar pathology score from the latest LFP window.

        Parameters
        ----------
        window : (window_length,) float64 — most recent samples, chronological

        Returns
        -------
        float — higher = more pathological.
                Compared against config.threshold: score >= threshold → trigger.
                The base class handles the comparison; subclasses just return
                the raw score.
        """
        ...
