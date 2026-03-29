"""Build a static trajectory dataset from the STN-GPe ODE simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import numpy as np
import yaml

from configs.sim_config import healthy_config, pathological_config
from src.simulation.runner import run_trajectory


@dataclass(frozen=True)
class BuildDatasetConfig:
    """Configuration for static trajectory dataset generation."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    regimes: tuple[str, ...] = ("healthy", "pathological")
    t_end_ms: float = 3000.0
    t_warmup_ms: float = 500.0
    include_state: bool = False
    output_dir: str = "artifacts/datasets/static_v1"


def _trajectory_id(regime: str, seed: int) -> str:
    return f"{regime}_seed{seed}"


def _make_regime_config(regime: str, t_end_ms: float, t_warmup_ms: float):
    if regime == "healthy":
        return healthy_config(t_end=t_end_ms, t_warmup=t_warmup_ms)
    if regime == "pathological":
        return pathological_config(t_end=t_end_ms, t_warmup=t_warmup_ms)
    raise ValueError(f"Unsupported regime: {regime}")


def build_static_dataset(cfg: BuildDatasetConfig) -> Path:
    """Generate trajectories, save NPZ files, and write a manifest CSV."""
    out_dir = Path(cfg.output_dir)
    traj_dir = out_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for regime in cfg.regimes:
        for seed in cfg.seeds:
            sim_cfg = _make_regime_config(
                regime=regime,
                t_end_ms=cfg.t_end_ms,
                t_warmup_ms=cfg.t_warmup_ms,
            )
            result = run_trajectory(config=sim_cfg, seed=seed)

            trajectory_id = _trajectory_id(regime=regime, seed=seed)
            rel_path = Path("trajectories") / f"{trajectory_id}.npz"
            abs_path = out_dir / rel_path

            save_payload = {
                "t": result["t"],
                "lfp": result["lfp"],
                "seed": np.array([seed], dtype=np.int64),
            }
            if cfg.include_state:
                save_payload["y"] = result["y"]
            np.savez_compressed(abs_path, **save_payload)

            t = np.asarray(result["t"])
            lfp = np.asarray(result["lfp"])
            dt_ms = float(np.mean(np.diff(t))) if len(t) > 1 else float("nan")

            manifest_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "regime": regime,
                    "seed": seed,
                    "fs_hz": float(sim_cfg.fs),
                    "t_end_ms": float(sim_cfg.t_end),
                    "t_warmup_ms": float(sim_cfg.t_warmup),
                    "n_samples": int(len(lfp)),
                    "dt_ms_mean": dt_ms,
                    "lfp_mean": float(np.mean(lfp)),
                    "lfp_std": float(np.std(lfp)),
                    "lfp_min": float(np.min(lfp)),
                    "lfp_max": float(np.max(lfp)),
                    "path": rel_path.as_posix(),
                }
            )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trajectory_id",
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
