"""
Open-loop stimulation sanity gate for the STN-GPe ODE model.

This module runs a frozen 5-seed gate to confirm the plant + stimulation path
behaves as intended before closed-loop controller integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import welch

from configs.sim_config import healthy_config, pathological_config
from .runner import run_trajectory


@dataclass(frozen=True)
class OpenLoopGateConfig:
    """Configuration for the frozen open-loop sanity gate."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    t_end_ms: float = 3000.0
    t_warmup_ms: float = 500.0
    pulse_frequency_hz: float = 130.0
    pulse_width_ms: float = 1.0
    weak_amplitude: float = 1.0
    strong_amplitude: float = 8.0
    beta_band_hz: tuple[float, float] = (13.0, 30.0)
    nperseg: int = 256
    min_seed_pass: int = 4


def make_pulse_train_stim(
    amplitude: float,
    frequency_hz: float = 130.0,
    pulse_width_ms: float = 1.0,
) -> callable:
    """Create a 0/amp pulse train stim function in milliseconds."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0")
    if pulse_width_ms <= 0.0:
        raise ValueError("pulse_width_ms must be > 0")

    # Convert frequency to period so the stim callback can run in ms.
    period_ms = 1000.0 / frequency_hz

    def stim_fn(t_ms: float) -> float:
        phase = t_ms % period_ms
        return amplitude if phase < pulse_width_ms else 0.0

    return stim_fn


def beta_power_welch(
    lfp: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float] = (13.0, 30.0),
    nperseg: int = 256,
) -> float:
    """Estimate beta-band power from Welch PSD and trapezoidal integration."""
    low_hz, high_hz = band_hz
    if low_hz >= high_hz:
        raise ValueError("beta band must satisfy low < high")

    nseg = min(int(nperseg), int(len(lfp)))
    if nseg <= 8:
        raise ValueError("lfp too short for stable PSD estimate")

    # Estimate PSD and integrate only the beta-band region.
    freqs, psd = welch(lfp, fs=fs_hz, nperseg=nseg)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _save_seed0_trace_plot(
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    names = [
        "healthy_no_stim",
        "pathological_no_stim",
        "pathological_weak_stim",
        "pathological_strong_stim",
    ]
    titles = [
        "Healthy / no stimulation",
        "Pathological / no stimulation",
        "Pathological / weak stimulation",
        "Pathological / strong stimulation",
    ]
    for ax, name, title in zip(axes, names, titles):
        t, lfp = traces[name]
        ax.plot(t, lfp, linewidth=0.8)
        ax.set_ylabel("LFP (mV)")
        ax.set_title(title)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_beta_summary_plot(rows: list[dict], output_path: Path) -> None:
    condition_order = [
        "healthy_no_stim",
        "pathological_no_stim",
        "pathological_weak_stim",
        "pathological_strong_stim",
    ]
    labels = [
        "healthy\nno stim",
        "pathological\nno stim",
        "pathological\nweak stim",
        "pathological\nstrong stim",
    ]

    means = []
    stds = []
    for condition in condition_order:
        vals = [r[condition] for r in rows]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set_ylabel("Beta power (13-30 Hz)")
    ax.set_title("Open-loop sanity gate: beta power by condition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_open_loop_sanity_gate(
    config: OpenLoopGateConfig,
    output_dir: str | Path,
) -> dict:
    """Run the frozen open-loop sanity gate and write artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weak_stim_fn = make_pulse_train_stim(
        amplitude=config.weak_amplitude,
        frequency_hz=config.pulse_frequency_hz,
        pulse_width_ms=config.pulse_width_ms,
    )
    strong_stim_fn = make_pulse_train_stim(
        amplitude=config.strong_amplitude,
        frequency_hz=config.pulse_frequency_hz,
        pulse_width_ms=config.pulse_width_ms,
    )

    rows: list[dict] = []
    seed0_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    errors: list[str] = []

    for seed in config.seeds:
        try:
            cfg_h = healthy_config(t_end=config.t_end_ms, t_warmup=config.t_warmup_ms)
            cfg_p = pathological_config(
                t_end=config.t_end_ms,
                t_warmup=config.t_warmup_ms,
            )

            # Run the four fixed open-loop conditions for each seed.
            run_h = run_trajectory(cfg_h, seed=seed)
            run_p0 = run_trajectory(cfg_p, seed=seed)
            run_pw = run_trajectory(cfg_p, seed=seed, stim_fn=weak_stim_fn)
            run_ps = run_trajectory(cfg_p, seed=seed, stim_fn=strong_stim_fn)

            row = {
                "seed": seed,
                "healthy_no_stim": beta_power_welch(
                    run_h["lfp"],
                    fs_hz=cfg_h.fs,
                    band_hz=config.beta_band_hz,
                    nperseg=config.nperseg,
                ),
                "pathological_no_stim": beta_power_welch(
                    run_p0["lfp"],
                    fs_hz=cfg_p.fs,
                    band_hz=config.beta_band_hz,
                    nperseg=config.nperseg,
                ),
                "pathological_weak_stim": beta_power_welch(
                    run_pw["lfp"],
                    fs_hz=cfg_p.fs,
                    band_hz=config.beta_band_hz,
                    nperseg=config.nperseg,
                ),
                "pathological_strong_stim": beta_power_welch(
                    run_ps["lfp"],
                    fs_hz=cfg_p.fs,
                    band_hz=config.beta_band_hz,
                    nperseg=config.nperseg,
                ),
            }

            patho = row["pathological_no_stim"]
            row["weak_ratio_vs_pathological"] = float(
                row["pathological_weak_stim"] / patho
            )
            row["strong_ratio_vs_pathological"] = float(
                row["pathological_strong_stim"] / patho
            )
            rows.append(row)

            if seed == config.seeds[0]:
                seed0_traces = {
                    "healthy_no_stim": (run_h["t"], run_h["lfp"]),
                    "pathological_no_stim": (run_p0["t"], run_p0["lfp"]),
                    "pathological_weak_stim": (run_pw["t"], run_pw["lfp"]),
                    "pathological_strong_stim": (run_ps["t"], run_ps["lfp"]),
                }
        except Exception as exc:  # defensive gate logging for long experiment loops
            errors.append(f"seed={seed}: {exc}")

    with (out_dir / "per_seed_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "seed",
            "healthy_no_stim",
            "pathological_no_stim",
            "pathological_weak_stim",
            "pathological_strong_stim",
            "weak_ratio_vs_pathological",
            "strong_ratio_vs_pathological",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    if seed0_traces:
        _save_seed0_trace_plot(seed0_traces, out_dir / "seed0_traces.png")
    if rows:
        _save_beta_summary_plot(rows, out_dir / "beta_summary.png")

    n_seeds = len(rows)
    # Majority-vote checks make the gate robust to a small number of noisy seeds.
    patho_gt_healthy = sum(
        r["pathological_no_stim"] > r["healthy_no_stim"] for r in rows
    )
    strong_suppresses = sum(
        r["pathological_strong_stim"] < r["pathological_no_stim"] for r in rows
    )
    dose_response = sum(
        r["strong_ratio_vs_pathological"] < r["weak_ratio_vs_pathological"]
        for r in rows
    )

    checks = {
        "no_runtime_errors": len(errors) == 0,
        "pathological_beta_gt_healthy_majority": patho_gt_healthy
        >= config.min_seed_pass,
        "strong_stim_suppresses_pathological_majority": strong_suppresses
        >= config.min_seed_pass,
        "dose_response_majority": dose_response >= config.min_seed_pass,
    }

    strong_ratios = [r["strong_ratio_vs_pathological"] for r in rows]
    weak_ratios = [r["weak_ratio_vs_pathological"] for r in rows]

    summary = {
        "gate_pass": bool(all(checks.values())),
        "config": {
            "seeds": list(config.seeds),
            "t_end_ms": config.t_end_ms,
            "t_warmup_ms": config.t_warmup_ms,
            "pulse_frequency_hz": config.pulse_frequency_hz,
            "pulse_width_ms": config.pulse_width_ms,
            "weak_amplitude": config.weak_amplitude,
            "strong_amplitude": config.strong_amplitude,
            "beta_band_hz": list(config.beta_band_hz),
            "nperseg": config.nperseg,
            "min_seed_pass": config.min_seed_pass,
        },
        "checks": checks,
        "counts": {
            "n_seeds_completed": n_seeds,
            "pathological_gt_healthy": int(patho_gt_healthy),
            "strong_suppresses": int(strong_suppresses),
            "dose_response": int(dose_response),
        },
        "aggregates": {
            "weak_ratio_mean": float(np.mean(weak_ratios))
            if weak_ratios
            else float("nan"),
            "weak_ratio_std": float(np.std(weak_ratios))
            if weak_ratios
            else float("nan"),
            "strong_ratio_mean": float(np.mean(strong_ratios))
            if strong_ratios
            else float("nan"),
            "strong_ratio_std": float(np.std(strong_ratios))
            if strong_ratios
            else float("nan"),
        },
        "errors": errors,
        "artifacts": {
            "per_seed_metrics_csv": str(out_dir / "per_seed_metrics.csv"),
            "seed0_trace_plot": str(out_dir / "seed0_traces.png"),
            "beta_summary_plot": str(out_dir / "beta_summary.png"),
            "summary_yaml": str(out_dir / "summary.yaml"),
        },
    }

    (out_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )
    return summary
