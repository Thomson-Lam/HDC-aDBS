"""Build static trajectory datasets (clean + transitional) from the ODE simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import numpy as np
import yaml

from configs.sim_config import healthy_config, pathological_config
from src.simulation.open_loop_sanity import make_pulse_train_stim
from src.simulation.runner import run_trajectory


@dataclass(frozen=True)
class BuildDatasetConfig:
    """Configuration for static trajectory dataset generation.

    Scenario meanings:
      - clean_healthy: baseline healthy trajectories
      - clean_pathological: baseline pathological trajectories
      - moderate: intermediate coupling trajectories (harder than clean)
      - onset: stitched healthy->pathological transition trajectories
      - recovery: pathological trajectories with stimulation turned off mid-run
      - healthy_holdout: long healthy-only runs reserved for specificity checks
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    scenarios: tuple[str, ...] = (
        "clean_healthy",
        "clean_pathological",
        "moderate",
        "onset",
        "recovery",
        "healthy_holdout",
    )
    t_end_ms: float = 3000.0
    t_warmup_ms: float = 500.0
    onset_segment_ms: float = 1500.0
    recovery_stim_off_ms: float | None = None
    recovery_stim_amplitude: float = 8.0
    pulse_frequency_hz: float = 130.0
    pulse_width_ms: float = 1.0
    moderate_g_gaba: float = 0.9
    healthy_holdout_t_end_ms: float = 6000.0
    transition_band_half_width_ms: float = 400.0
    include_state: bool = False
    output_dir: str = "artifacts/datasets/static_v1"


def _trajectory_id(scenario: str, seed: int) -> str:
    return f"{scenario}_seed{seed}"


def _resolved_recovery_stim_off_ms(cfg: BuildDatasetConfig) -> float:
    if cfg.recovery_stim_off_ms is not None:
        return float(cfg.recovery_stim_off_ms)
    # Default recovery transition to the middle of the usable (post-warmup) span.
    return float(cfg.t_warmup_ms + 0.5 * (cfg.t_end_ms - cfg.t_warmup_ms))


def _validate_build_config(cfg: BuildDatasetConfig) -> None:
    allowed_scenarios = {
        "clean_healthy",
        "clean_pathological",
        "moderate",
        "onset",
        "recovery",
        "healthy_holdout",
    }

    if len(cfg.seeds) == 0:
        raise ValueError("seeds must be non-empty")
    if len(cfg.scenarios) == 0:
        raise ValueError("scenarios must be non-empty")
    unknown = set(cfg.scenarios) - allowed_scenarios
    if unknown:
        raise ValueError(f"Unsupported scenario(s): {sorted(unknown)}")

    if cfg.t_end_ms <= cfg.t_warmup_ms:
        raise ValueError("t_end_ms must be greater than t_warmup_ms")
    if cfg.onset_segment_ms <= cfg.t_warmup_ms:
        raise ValueError("onset_segment_ms must be greater than t_warmup_ms")
    if cfg.healthy_holdout_t_end_ms <= cfg.t_warmup_ms:
        raise ValueError("healthy_holdout_t_end_ms must be greater than t_warmup_ms")

    if cfg.pulse_frequency_hz <= 0.0:
        raise ValueError("pulse_frequency_hz must be > 0")
    if cfg.pulse_width_ms <= 0.0:
        raise ValueError("pulse_width_ms must be > 0")

    if cfg.transition_band_half_width_ms <= 0.0:
        raise ValueError("transition_band_half_width_ms must be > 0")

    if "recovery" in set(cfg.scenarios):
        stim_off = _resolved_recovery_stim_off_ms(cfg)
        if stim_off <= cfg.t_warmup_ms:
            raise ValueError(
                "recovery_stim_off_ms must be greater than t_warmup_ms "
                f"(got stim_off={stim_off}, warmup={cfg.t_warmup_ms})"
            )
        if stim_off >= cfg.t_end_ms:
            raise ValueError(
                "recovery_stim_off_ms must be less than t_end_ms "
                f"(got stim_off={stim_off}, t_end={cfg.t_end_ms})"
            )

        aligned = stim_off - cfg.t_warmup_ms
        post_warmup_duration = cfg.t_end_ms - cfg.t_warmup_ms
        if aligned <= cfg.transition_band_half_width_ms:
            raise ValueError(
                "recovery transition occurs too early after warmup for labeled windows; "
                f"aligned={aligned} ms, half_width={cfg.transition_band_half_width_ms} ms"
            )
        if aligned >= (post_warmup_duration - cfg.transition_band_half_width_ms):
            raise ValueError(
                "recovery transition occurs too late in trajectory for labeled windows; "
                f"aligned={aligned} ms, post_warmup_duration={post_warmup_duration} ms, "
                f"half_width={cfg.transition_band_half_width_ms} ms"
            )


def _make_regime_config(
    regime: str, t_end_ms: float, t_warmup_ms: float, g_gaba: float | None = None
):
    if regime == "healthy":
        return healthy_config(t_end=t_end_ms, t_warmup=t_warmup_ms)
    if regime == "pathological":
        cfg = pathological_config(t_end=t_end_ms, t_warmup=t_warmup_ms)
        if g_gaba is not None:
            cfg.g_GABA = float(g_gaba)
        return cfg
    raise ValueError(f"Unsupported regime: {regime}")


def _normalize_timebase(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return t
    return t - float(t[0])


def _build_clean_trajectory(
    regime: str, seed: int, cfg: BuildDatasetConfig
) -> dict[str, object]:
    sim_cfg = _make_regime_config(
        regime=regime, t_end_ms=cfg.t_end_ms, t_warmup_ms=cfg.t_warmup_ms
    )
    result = run_trajectory(config=sim_cfg, seed=seed)
    t = _normalize_timebase(result["t"])
    lfp = np.asarray(result["lfp"], dtype=np.float64)
    scenario = f"clean_{regime}"
    label = 0 if regime == "healthy" else 1
    return {
        "scenario": scenario,
        "regime": regime,
        "t": t,
        "lfp": lfp,
        "y": result.get("y"),
        "fs_hz": float(sim_cfg.fs),
        "label_strategy": "constant",
        "default_label": int(label),
        "transition_time_ms": float("nan"),
        "transition_band_half_width_ms": float("nan"),
    }


def _build_moderate_trajectory(seed: int, cfg: BuildDatasetConfig) -> dict[str, object]:
    sim_cfg = _make_regime_config(
        regime="pathological",
        t_end_ms=cfg.t_end_ms,
        t_warmup_ms=cfg.t_warmup_ms,
        g_gaba=cfg.moderate_g_gaba,
    )
    result = run_trajectory(config=sim_cfg, seed=seed)
    t = _normalize_timebase(result["t"])
    lfp = np.asarray(result["lfp"], dtype=np.float64)
    return {
        "scenario": "moderate",
        "regime": "pathological",
        "t": t,
        "lfp": lfp,
        "y": result.get("y"),
        "fs_hz": float(sim_cfg.fs),
        "label_strategy": "constant",
        "default_label": 1,
        "transition_time_ms": float("nan"),
        "transition_band_half_width_ms": float("nan"),
    }


def _build_onset_trajectory(seed: int, cfg: BuildDatasetConfig) -> dict[str, object]:
    # Build onset by stitching a healthy segment followed by a pathological segment.
    healthy_seg = run_trajectory(
        config=healthy_config(t_end=cfg.onset_segment_ms, t_warmup=cfg.t_warmup_ms),
        seed=seed,
    )
    path_seg = run_trajectory(
        config=pathological_config(
            t_end=cfg.onset_segment_ms, t_warmup=cfg.t_warmup_ms
        ),
        seed=seed,
    )

    t_h = _normalize_timebase(healthy_seg["t"])
    t_p = _normalize_timebase(path_seg["t"])
    dt = float(np.mean(np.diff(t_h))) if len(t_h) > 1 else 1000.0 / 250.0
    transition_time = float(t_h[-1] + dt) if len(t_h) > 0 else 0.0

    t = np.concatenate([t_h, t_p + transition_time])
    lfp = np.concatenate(
        [
            np.asarray(healthy_seg["lfp"], dtype=np.float64),
            np.asarray(path_seg["lfp"], dtype=np.float64),
        ]
    )

    return {
        "scenario": "onset",
        "regime": "transition",
        "t": t,
        "lfp": lfp,
        "y": None,
        "fs_hz": 250.0,
        "label_strategy": "before_after_transition",
        "default_label": -1,
        "transition_time_ms": transition_time,
        "transition_band_half_width_ms": float(cfg.transition_band_half_width_ms),
    }


def _build_recovery_trajectory(seed: int, cfg: BuildDatasetConfig) -> dict[str, object]:
    stim_off_ms = _resolved_recovery_stim_off_ms(cfg)
    pulse_fn = make_pulse_train_stim(
        amplitude=cfg.recovery_stim_amplitude,
        frequency_hz=cfg.pulse_frequency_hz,
        pulse_width_ms=cfg.pulse_width_ms,
    )

    # Stimulation is on before stim_off, then turned fully off for recovery phase.
    def recovery_stim_fn(t_ms: float) -> float:
        if t_ms < stim_off_ms:
            return float(pulse_fn(t_ms))
        return 0.0

    sim_cfg = pathological_config(t_end=cfg.t_end_ms, t_warmup=cfg.t_warmup_ms)
    result = run_trajectory(config=sim_cfg, seed=seed, stim_fn=recovery_stim_fn)

    t = _normalize_timebase(result["t"])
    lfp = np.asarray(result["lfp"], dtype=np.float64)
    stim_off_aligned = max(0.0, float(stim_off_ms - cfg.t_warmup_ms))

    return {
        "scenario": "recovery",
        "regime": "transition",
        "t": t,
        "lfp": lfp,
        "y": result.get("y"),
        "fs_hz": float(sim_cfg.fs),
        "label_strategy": "before_after_transition",
        "default_label": -1,
        "transition_time_ms": stim_off_aligned,
        "transition_band_half_width_ms": float(cfg.transition_band_half_width_ms),
    }


def _build_healthy_holdout_trajectory(
    seed: int, cfg: BuildDatasetConfig
) -> dict[str, object]:
    sim_cfg = healthy_config(
        t_end=cfg.healthy_holdout_t_end_ms,
        t_warmup=cfg.t_warmup_ms,
    )
    result = run_trajectory(config=sim_cfg, seed=seed)
    t = _normalize_timebase(result["t"])
    lfp = np.asarray(result["lfp"], dtype=np.float64)
    return {
        "scenario": "healthy_holdout",
        "regime": "healthy",
        "t": t,
        "lfp": lfp,
        "y": result.get("y"),
        "fs_hz": float(sim_cfg.fs),
        "label_strategy": "constant",
        "default_label": 0,
        "transition_time_ms": float("nan"),
        "transition_band_half_width_ms": float("nan"),
    }


def build_static_dataset(cfg: BuildDatasetConfig) -> Path:
    """Generate trajectories, save NPZ files, and write a manifest CSV."""
    _validate_build_config(cfg)
    out_dir = Path(cfg.output_dir)
    traj_dir = out_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for scenario in cfg.scenarios:
        for seed in cfg.seeds:
            # Dispatch each requested scenario to its trajectory generator.
            if scenario == "clean_healthy":
                payload = _build_clean_trajectory("healthy", seed, cfg)
            elif scenario == "clean_pathological":
                payload = _build_clean_trajectory("pathological", seed, cfg)
            elif scenario == "moderate":
                payload = _build_moderate_trajectory(seed, cfg)
            elif scenario == "onset":
                payload = _build_onset_trajectory(seed, cfg)
            elif scenario == "recovery":
                payload = _build_recovery_trajectory(seed, cfg)
            elif scenario == "healthy_holdout":
                payload = _build_healthy_holdout_trajectory(seed, cfg)
            else:
                raise ValueError(f"Unsupported scenario: {scenario}")

            trajectory_id = _trajectory_id(scenario=scenario, seed=seed)
            rel_path = Path("trajectories") / f"{trajectory_id}.npz"
            abs_path = out_dir / rel_path

            t = np.asarray(payload["t"], dtype=np.float64)
            lfp = np.asarray(payload["lfp"], dtype=np.float64)
            dt_ms = float(np.mean(np.diff(t))) if len(t) > 1 else float("nan")
            save_payload = {
                "t": t,
                "lfp": lfp,
                "seed": np.array([seed], dtype=np.int64),
            }
            if cfg.include_state and payload["y"] is not None:
                save_payload["y"] = payload["y"]
            np.savez_compressed(abs_path, **save_payload)

            # Keep one manifest row per trajectory so downstream split/window code is reproducible.
            manifest_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "scenario": str(payload["scenario"]),
                    "regime": str(payload["regime"]),
                    "seed": seed,
                    "fs_hz": float(payload["fs_hz"]),
                    "t_end_ms": float(t[-1]) if len(t) > 0 else 0.0,
                    "t_warmup_ms": float(cfg.t_warmup_ms),
                    "n_samples": int(len(lfp)),
                    "dt_ms_mean": dt_ms,
                    "lfp_mean": float(np.mean(lfp)),
                    "lfp_std": float(np.std(lfp)),
                    "lfp_min": float(np.min(lfp)),
                    "lfp_max": float(np.max(lfp)),
                    "path": rel_path.as_posix(),
                    "label_strategy": str(payload["label_strategy"]),
                    "default_label": int(payload["default_label"]),
                    "transition_time_ms": float(payload["transition_time_ms"]),
                    "transition_band_half_width_ms": float(
                        payload["transition_band_half_width_ms"]
                    ),
                }
            )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trajectory_id",
            "scenario",
            "regime",
            "seed",
            "fs_hz",
            "t_end_ms",
            "t_warmup_ms",
            "n_samples",
            "dt_ms_mean",
            "lfp_mean",
            "lfp_std",
            "lfp_min",
            "lfp_max",
            "path",
            "label_strategy",
            "default_label",
            "transition_time_ms",
            "transition_band_half_width_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    build_cfg_path = out_dir / "build_config.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
    )

    return out_dir
