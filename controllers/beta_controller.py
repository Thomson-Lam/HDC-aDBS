"""
Classical beta-power baseline controller.

Uses a causal 2nd-order Butterworth bandpass filter (13–30 Hz) followed by
RMS power estimation to detect pathological beta bursts.

Key design constraint: NO non-causal filtering.  The experiment spec requires
that all controllers see the same causal signal path so the comparison with
HDC is fair.  We achieve this by:
  - Using second-order sections (SOS) for numerical stability.
  - Carrying the IIR filter state (_zi) across chunk boundaries so the filter
    is globally causal even though we receive data in short chunks.
  - Overriding ingest() to run the filter before buffering, ensuring
    _compute_decision() always sees the filtered (beta-band) signal.

Decision pipeline (per window):
    raw LFP chunk (mV) → causal IIR bandpass (13–30 Hz, state carried)
        → filtered chunk appended to ring buffer
        → RMS of latest 128-sample window
        → compared against config.threshold (units: mV RMS)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .base import BaseController, ControllerConfig


class BetaController(BaseController):
    """Baseline controller: causal beta-band RMS thresholding.

    Parameters
    ----------
    config       : ControllerConfig — timing and threshold parameters.
                   threshold is interpreted as raw RMS in mV.
    beta_low_hz  : float — lower edge of beta band (default 13 Hz).
    beta_high_hz : float — upper edge of beta band (default 30 Hz).
    filter_order : int   — Butterworth filter order (default 2).
                   Higher order → steeper roll-off but more group delay.
                   2nd order is a common clinical standard.
    """

    def __init__(
        self,
        config: ControllerConfig | None = None,
        beta_low_hz: float = 13.0,
        beta_high_hz: float = 30.0,
        filter_order: int = 2,
    ) -> None:
        super().__init__(config or ControllerConfig())
        self.beta_low_hz  = beta_low_hz
        self.beta_high_hz = beta_high_hz
        self.filter_order = filter_order

        # Build second-order sections (SOS) representation of the bandpass filter.
        # SOS is preferred over transfer-function (b, a) form because it avoids
        # numerical precision issues for higher-order filters.
        nyq = self.config.fs / 2.0
        self._sos: np.ndarray = butter(
            filter_order,
            [beta_low_hz / nyq, beta_high_hz / nyq],
            btype='bandpass',
            output='sos',
        )

        # Initial filter state for sosfilt (shape: (n_sections, 2)).
        # sosfilt_zi returns unit-DC-gain initial conditions; we start with
        # zero signal, so the actual initial state is all zeros — sosfilt_zi
        # * 0 = zeros, but we call sosfilt_zi once to get the right shape.
        self._zi: np.ndarray = np.zeros_like(sosfilt_zi(self._sos))

    # ------------------------------------------------------------------
    # Override ingest to filter before buffering
    # ------------------------------------------------------------------

    def ingest(self, t: np.ndarray, lfp: np.ndarray) -> None:
        """Filter the incoming LFP chunk, then delegate to BaseController.

        The causal IIR filter is applied chunk-by-chunk here.  The filter
        state _zi is updated so the next chunk continues from where this one
        left off, preserving global causality across chunk boundaries.

        After filtering, the filtered signal is passed to super().ingest()
        which handles the ring buffer, state machine, and decision logic.
        The ring buffer therefore holds filtered (beta-band) samples, which
        is what _compute_decision() expects.
        """
        # Apply the bandpass filter to this chunk, updating the IIR state.
        # sosfilt returns (filtered_output, updated_state).
        filtered_lfp, self._zi = sosfilt(self._sos, lfp, zi=self._zi)

        # Pass the filtered signal up to the base class for buffering and
        # decision logic.  The base class sees only the filtered signal.
        super().ingest(t, filtered_lfp)

    def reset(self) -> None:
        """Reset buffer, state machine, metrics, AND filter state."""
        super().reset()
        # Zero the IIR state so the filter starts fresh on the next run.
        self._zi = np.zeros_like(sosfilt_zi(self._sos))

    # ------------------------------------------------------------------
    # Decision: RMS of the (already filtered) window
    # ------------------------------------------------------------------

    def _compute_decision(self, window: np.ndarray) -> float:
        """Return the RMS of the beta-band signal in the current window.

        The window was drawn from the ring buffer which already holds
        filtered samples (see ingest() above), so no additional filtering
        is needed here.

        Parameters
        ----------
        window : (window_length,) float64 — filtered LFP, mV

        Returns
        -------
        float — RMS amplitude in mV.  Higher = more beta power = more pathological.
                Compared against config.threshold; units are mV.
        """
        return float(np.sqrt(np.mean(window ** 2)))
