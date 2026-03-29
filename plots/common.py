"""Shared plotting helpers for artifact-driven reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def set_plot_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.5),
            "figure.dpi": 130,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_yaml_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_current(fig_path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return fig_path
