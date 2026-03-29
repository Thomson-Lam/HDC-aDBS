"""Run a reproducible four-condition closed-loop benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import yaml

from configs.sim_config import healthy_config, pathological_config
from src.simulation.runner import run_trajectory

from .base import ControllerConfig
from .beta_controller import BetaController
from .eval_metrics import (
    beta_envelope_rms,
    decision_time_stats,
    duty_cycle_from_stim,
    healthy_false_trigger_rate_per_minute,
    mean_beta_power,
    memory_stats_bytes,
    pathological_occupancy,
    pulse_count_from_stim,
    suppression_latency_ms,
)
from .hdc_controller import HDCController
from .run_controller import run_closed_loop
from .waveform import make_epoch_gated_pulse_train


METRIC_DEFINITIONS_VERSION = "experiment-spec-v1"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _condition_rows_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["regime"])].append(row)

    metrics = [
        "mean_beta_power",
        "pathological_occupancy",
        "suppression_latency_ms",
        "duty_cycle",
        "pulse_count",
        "decision_time_mean_ms",
        "decision_time_p95_ms",
        "healthy_false_trigger_rate_per_min",
        "unnecessary_stim_duty_cycle",
        "pulses_per_min_healthy",
    ]
    out: list[dict] = []
    for (condition, regime), group in grouped.items():
        entry: dict[str, float | str | int] = {
            "condition": condition,
            "regime": regime,
            "n_runs": len(group),
        }
        for metric in metrics:
            vals = np.asarray(
                [float(g[metric]) for g in group if g.get(metric) is not None],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            entry[f"{metric}_mean"] = float(vals.mean()) if vals.size else float("nan")
            entry[f"{metric}_std"] = float(vals.std()) if vals.size else float("nan")
        out.append(entry)
    return sorted(out, key=lambda x: (str(x["regime"]), str(x["condition"])))


def _make_shared_controller_config(args: argparse.Namespace) -> ControllerConfig:
    return ControllerConfig(
        fs=250.0,
        window_length=128,
        decision_cadence_ms=50.0,
        stim_duration_ms=200.0,
        lockout_duration_ms=200.0,
        stim_amplitude=args.stim_amplitude,
        pulse_frequency_hz=130.0,
        pulse_width_ms=1.0,
    )


def _compute_run_metrics(
    *,
    t_ms: np.ndarray,
    lfp: np.ndarray,
    stim: np.ndarray,
    stim_onsets_ms: list[float],
    decision_times_ms: list[float],
    pathology_threshold: float,
    fs_hz: float,
) -> dict:
    beta_env = beta_envelope_rms(lfp, fs_hz=fs_hz)
    dec = decision_time_stats(decision_times_ms)
    duration_ms = float(t_ms[-1] - t_ms[0]) if t_ms.size else 0.0
    pulse_count = pulse_count_from_stim(stim)
    return {
        "duration_ms": duration_ms,
        "mean_beta_power": mean_beta_power(lfp, fs_hz=fs_hz),
        "pathological_occupancy": pathological_occupancy(beta_env, pathology_threshold),
        "suppression_latency_ms": suppression_latency_ms(
            t_ms=t_ms,
            beta_envelope=beta_env,
            threshold=pathology_threshold,
            stim_onset_times_ms=stim_onsets_ms,
        ),
        "duty_cycle": duty_cycle_from_stim(stim),
        "pulse_count": pulse_count,
        "decision_time_mean_ms": dec["mean_ms"],
        "decision_time_p95_ms": dec["p95_ms"],
        "pulses_per_min": (
            float(pulse_count / (duration_ms / 60000.0)) if duration_ms > 0.0 else 0.0
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    healthy_seeds = [int(s) for s in args.healthy_seeds.split(",") if s.strip()]
    cfg_template = _make_shared_controller_config(args)

    # First pass: no-stim pathological only, to lock occupancy threshold.
    patho_nostim_cache: dict[int, dict] = {}
    for seed in seeds:
        cfg = pathological_config(t_end=args.t_end_ms, t_warmup=args.t_warmup_ms)
        run = run_trajectory(cfg, seed=seed)
        patho_nostim_cache[seed] = run

    occupancy_ref = np.concatenate(
        [
            beta_envelope_rms(run["lfp"], fs_hz=250.0)
            for run in patho_nostim_cache.values()
        ]
    )
    pathology_threshold = float(np.percentile(occupancy_ref, args.pathology_percentile))

    rows: list[dict] = []

    # 1) No stimulation
    for seed in seeds:
        run = patho_nostim_cache[seed]
        stim = np.zeros_like(run["t"], dtype=np.float64)
        metrics = _compute_run_metrics(
            t_ms=run["t"],
            lfp=run["lfp"],
            stim=stim,
            stim_onsets_ms=[],
            decision_times_ms=[],
            pathology_threshold=pathology_threshold,
            fs_hz=250.0,
        )
        rows.append(
            {
                "seed": seed,
                "condition": "no_stimulation",
                "regime": "pathological",
                **metrics,
                "healthy_false_trigger_rate_per_min": float("nan"),
                "unnecessary_stim_duty_cycle": float("nan"),
                "pulses_per_min_healthy": float("nan"),
            }
        )

    # 2) Continuous DBS with identical pulse waveform always active
    for seed in seeds:
        cfg = pathological_config(t_end=args.t_end_ms, t_warmup=args.t_warmup_ms)
        stim_fn = make_epoch_gated_pulse_train(
            amplitude=cfg_template.stim_amplitude,
            frequency_hz=cfg_template.pulse_frequency_hz,
            pulse_width_ms=cfg_template.pulse_width_ms,
            epoch_active_fn=lambda _t: True,
        )
        run = run_trajectory(cfg, seed=seed, stim_fn=stim_fn)
        stim = np.asarray([stim_fn(float(t_i)) for t_i in run["t"]], dtype=np.float64)
        metrics = _compute_run_metrics(
            t_ms=run["t"],
            lfp=run["lfp"],
            stim=stim,
            stim_onsets_ms=[float(run["t"][0])] if len(run["t"]) else [],
            decision_times_ms=[],
            pathology_threshold=pathology_threshold,
            fs_hz=250.0,
        )
        rows.append(
            {
                "seed": seed,
                "condition": "continuous_dbs",
                "regime": "pathological",
                **metrics,
                "healthy_false_trigger_rate_per_min": float("nan"),
                "unnecessary_stim_duty_cycle": float("nan"),
                "pulses_per_min_healthy": float("nan"),
            }
        )

    # 3) Beta-threshold aDBS
    for seed in seeds:
        cfg = pathological_config(t_end=args.t_end_ms, t_warmup=args.t_warmup_ms)
        ctrl_cfg = ControllerConfig(
            **{**asdict(cfg_template), "threshold": args.beta_threshold}
        )
        ctrl = BetaController(config=ctrl_cfg)
        run = run_closed_loop(ctrl, cfg, seed=seed)
        stim_onsets = list(run["metrics"].stim_times_ms)
        metrics = _compute_run_metrics(
            t_ms=run["t"],
            lfp=run["lfp"],
            stim=run["stim"],
            stim_onsets_ms=stim_onsets,
            decision_times_ms=list(run["metrics"].decision_times_ms),
            pathology_threshold=pathology_threshold,
            fs_hz=250.0,
        )
        rows.append(
            {
                "seed": seed,
                "condition": "beta_adbs",
                "regime": "pathological",
                **metrics,
                "healthy_false_trigger_rate_per_min": float("nan"),
                "unnecessary_stim_duty_cycle": float("nan"),
                "pulses_per_min_healthy": float("nan"),
            }
        )

    # 4) HDC-triggered aDBS
    for seed in seeds:
        cfg = pathological_config(t_end=args.t_end_ms, t_warmup=args.t_warmup_ms)
        ctrl_cfg = ControllerConfig(
            **{**asdict(cfg_template), "threshold": args.hdc_threshold}
        )
        ctrl = HDCController(
            model_path=args.model_path,
            config=ctrl_cfg,
            trainer_type=args.trainer_type,
        )
        run = run_closed_loop(ctrl, cfg, seed=seed)
        stim_onsets = list(run["metrics"].stim_times_ms)
        metrics = _compute_run_metrics(
            t_ms=run["t"],
            lfp=run["lfp"],
            stim=run["stim"],
            stim_onsets_ms=stim_onsets,
            decision_times_ms=list(run["metrics"].decision_times_ms),
            pathology_threshold=pathology_threshold,
            fs_hz=250.0,
        )
        rows.append(
            {
                "seed": seed,
                "condition": "hdc_adbs",
                "regime": "pathological",
                **metrics,
                "healthy_false_trigger_rate_per_min": float("nan"),
                "unnecessary_stim_duty_cycle": float("nan"),
                "pulses_per_min_healthy": float("nan"),
            }
        )

    # Healthy specificity runs for adaptive controllers only.
    for condition in ("beta_adbs", "hdc_adbs"):
        for seed in healthy_seeds:
            cfg = healthy_config(t_end=args.t_end_ms, t_warmup=args.t_warmup_ms)
            if condition == "beta_adbs":
                ctrl_cfg = ControllerConfig(
                    **{**asdict(cfg_template), "threshold": args.beta_threshold}
                )
                ctrl = BetaController(config=ctrl_cfg)
            else:
                ctrl_cfg = ControllerConfig(
                    **{**asdict(cfg_template), "threshold": args.hdc_threshold}
                )
                ctrl = HDCController(
                    model_path=args.model_path,
                    config=ctrl_cfg,
                    trainer_type=args.trainer_type,
                )
            run = run_closed_loop(ctrl, cfg, seed=seed)
            stim = run["stim"]
            duration_ms = float(run["t"][-1] - run["t"][0]) if len(run["t"]) else 0.0
            pulse_count = pulse_count_from_stim(stim)
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "regime": "healthy",
                    "duration_ms": duration_ms,
                    "mean_beta_power": mean_beta_power(run["lfp"], fs_hz=250.0),
                    "pathological_occupancy": float("nan"),
                    "suppression_latency_ms": float("nan"),
                    "duty_cycle": duty_cycle_from_stim(stim),
                    "pulse_count": pulse_count,
                    "pulses_per_min": (
                        float(pulse_count / (duration_ms / 60000.0))
                        if duration_ms > 0.0
                        else 0.0
                    ),
                    "decision_time_mean_ms": decision_time_stats(
                        run["metrics"].decision_times_ms
                    )["mean_ms"],
                    "decision_time_p95_ms": decision_time_stats(
                        run["metrics"].decision_times_ms
                    )["p95_ms"],
                    "healthy_false_trigger_rate_per_min": healthy_false_trigger_rate_per_minute(
                        n_stimulations=run["metrics"].n_stimulations,
                        duration_ms=duration_ms,
                    ),
                    "unnecessary_stim_duty_cycle": duty_cycle_from_stim(stim),
                    "pulses_per_min_healthy": (
                        float(pulse_count / (duration_ms / 60000.0))
                        if duration_ms > 0.0
                        else 0.0
                    ),
                }
            )

    per_run_path = out_dir / "per_run_metrics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with per_run_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = _condition_rows_summary(rows)
    summary_csv_path = out_dir / "summary_by_condition.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted({k for row in summary_rows for k in row.keys()})
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "seeds": seeds,
        "healthy_seeds": healthy_seeds,
        "git_commit": _git_commit(),
        "controller_config": asdict(cfg_template),
        "thresholds": {
            "beta_threshold": args.beta_threshold,
            "hdc_threshold": args.hdc_threshold,
            "pathology_threshold_percentile": args.pathology_percentile,
            "pathology_threshold_value": pathology_threshold,
        },
        "python": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (out_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(run_manifest, sort_keys=False),
        encoding="utf-8",
    )

    summary = {
        "conditions": ["no_stimulation", "continuous_dbs", "beta_adbs", "hdc_adbs"],
        "summary_by_condition": summary_rows,
        "artifacts": {
            "per_run_metrics_csv": str(per_run_path),
            "summary_by_condition_csv": str(summary_csv_path),
            "run_manifest_yaml": str(out_dir / "run_manifest.yaml"),
        },
        "memory_stats": memory_stats_bytes(
            {"rows": rows, "summary_rows": summary_rows}
        ),
    }
    (out_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="artifacts/models/linear")
    p.add_argument("--trainer-type", choices=["linear", "prototype"], default="linear")
    p.add_argument("--output-dir", default="artifacts/closed_loop")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--healthy-seeds", default="100,101,102,103,104")
    p.add_argument("--t-end-ms", type=float, default=3000.0)
    p.add_argument("--t-warmup-ms", type=float, default=500.0)
    p.add_argument("--stim-amplitude", type=float, default=1.5)
    p.add_argument("--beta-threshold", type=float, default=0.0)
    p.add_argument("--hdc-threshold", type=float, default=0.0)
    p.add_argument("--pathology-percentile", type=float, default=50.0)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_benchmark(args)
    print(json.dumps(summary["artifacts"], indent=2))


if __name__ == "__main__":
    main()
