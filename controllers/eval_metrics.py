"""Pure metric functions for closed-loop controller benchmarking."""

from __future__ import annotations

import sys
from collections.abc import Mapping

import numpy as np
from scipy.signal import butter, sosfilt, welch


def mean_beta_power(
    lfp: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float] = (13.0, 30.0),
    nperseg: int = 256,
) -> float:
    """Compute mean beta-band power from Welch PSD."""
    low_hz, high_hz = band_hz
    nseg = min(int(nperseg), int(len(lfp)))
    if nseg <= 8:
        return 0.0
    freqs, psd = welch(lfp, fs=fs_hz, nperseg=nseg)
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return 0.0
    return float(np.trapezoid(psd[band_mask], freqs[band_mask]))


def beta_envelope_rms(
    lfp: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float] = (13.0, 30.0),
    rms_window_samples: int = 128,
) -> np.ndarray:
    """Return causal beta-band RMS envelope."""
    nyq = fs_hz / 2.0
    sos = butter(
        2, [band_hz[0] / nyq, band_hz[1] / nyq], btype="bandpass", output="sos"
    )
    beta = sosfilt(sos, lfp)
    sq = beta * beta
    kernel = np.ones(max(1, int(rms_window_samples)), dtype=np.float64)
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") / float(kernel.size))
    return rms


def pathological_occupancy(beta_envelope: np.ndarray, threshold: float) -> float:
    """Fraction of samples where beta envelope exceeds pathological threshold."""
    if beta_envelope.size == 0:
        return 0.0
    return float(np.mean(beta_envelope >= threshold))


def suppression_latency_ms(
    t_ms: np.ndarray,
    beta_envelope: np.ndarray,
    threshold: float,
    stim_onset_times_ms: list[float],
    sustained_duration_ms: float = 100.0,
) -> float:
    """Latency from first stimulation onset to sustained recovery below threshold."""
    if len(stim_onset_times_ms) == 0 or t_ms.size == 0:
        return float("nan")

    t0 = float(stim_onset_times_ms[0])
    start_idx = int(np.searchsorted(t_ms, t0, side="left"))
    if start_idx >= t_ms.size:
        return float("nan")

    below = beta_envelope[start_idx:] < threshold
    if not np.any(below):
        return float("nan")

    if t_ms.size < 2:
        return float("nan")
    dt_ms = float(np.median(np.diff(t_ms)))
    need = max(1, int(round(sustained_duration_ms / dt_ms)))

    run = 0
    for i, ok in enumerate(below):
        run = run + 1 if ok else 0
        if run >= need:
            hit_idx = start_idx + i - need + 1
            return float(t_ms[hit_idx] - t0)
    return float("nan")


def duty_cycle_from_stim(stim: np.ndarray) -> float:
    """Fraction of samples with non-zero stimulation."""
    if stim.size == 0:
        return 0.0
    return float(np.mean(stim > 0.0))


def pulse_count_from_stim(stim: np.ndarray) -> int:
    """Count pulse onsets from sampled stimulation waveform."""
    if stim.size == 0:
        return 0
    on = stim > 0.0
    starts = on & np.concatenate((np.array([True]), ~on[:-1]))
    return int(np.sum(starts))


def healthy_false_trigger_rate_per_minute(
    n_stimulations: int,
    duration_ms: float,
) -> float:
    """False-trigger rate on healthy trajectories in stim epochs / minute."""
    if duration_ms <= 0.0:
        return 0.0
    return float(n_stimulations / (duration_ms / 60000.0))


def decision_time_stats(decision_times_ms: list[float]) -> dict[str, float]:
    """Return mean/p95/max decision compute times."""
    x = np.asarray(decision_times_ms, dtype=np.float64)
    if x.size == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(x.mean()),
        "p95_ms": float(np.percentile(x, 95.0)),
        "max_ms": float(x.max()),
    }


def memory_stats_bytes(objects: Mapping[str, object]) -> dict[str, int]:
    """Lightweight memory estimation for selected Python objects."""
    out: dict[str, int] = {}
    total = 0
    for name, obj in objects.items():
        size = int(sys.getsizeof(obj))
        out[name] = size
        total += size
    out["total_bytes"] = total
    return out
