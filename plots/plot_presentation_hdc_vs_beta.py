from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plots.common import ARTIFACTS, read_csv_if_exists, save_current


def _pct_delta(old: float, new: float) -> float:
    if old == 0.0:
        return 0.0
    return ((new - old) / old) * 100.0


def generate(output_dir: Path) -> Path | None:
    df = read_csv_if_exists(ARTIFACTS / "closed_loop" / "summary_by_condition.csv")
    if df is None or df.empty:
        print("[skip] summary_by_condition.csv not found (run closed-loop benchmark first)")
        return None

    patho = (
        df[df["regime"] == "pathological"]
        .set_index("condition")
        .reindex(["beta_adbs", "hdc_adbs"])
    )
    healthy = (
        df[df["regime"] == "healthy"]
        .set_index("condition")
        .reindex(["beta_adbs", "hdc_adbs"])
    )

    needed_patho = {"pathological_occupancy_mean", "duty_cycle_mean"}
    needed_healthy = {"healthy_false_trigger_rate_per_min_mean"}
    if not needed_patho.issubset(patho.columns) or not needed_healthy.issubset(
        healthy.columns
    ):
        print("[skip] required presentation metrics missing from summary_by_condition.csv")
        return None

    labels = ["Beta aDBS", "HDC aDBS"]
    colors = ["#d95f02", "#1b9e77"]

    occ = patho["pathological_occupancy_mean"].to_numpy(dtype=np.float64)
    duty = patho["duty_cycle_mean"].to_numpy(dtype=np.float64)
    false_triggers = healthy["healthy_false_trigger_rate_per_min_mean"].to_numpy(
        dtype=np.float64
    )

    occ_change = _pct_delta(float(occ[0]), float(occ[1]))
    duty_change = _pct_delta(float(duty[0]), float(duty[1]))
    ftr_change = _pct_delta(float(false_triggers[0]), float(false_triggers[1]))

    plt.rcParams.update(
        {
            "figure.figsize": (12.5, 6.8),
            "figure.dpi": 140,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(1, 3)
    fig.patch.set_facecolor("#f6f3ea")

    for ax in axes:
        ax.set_facecolor("#fffdf8")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    panels = [
        (
            axes[0],
            occ,
            "Pathological Occupancy",
            "Fraction of pathological beta-envelope samples",
            occ_change,
        ),
        (
            axes[1],
            duty,
            "Duty Cycle",
            "Fraction of time stimulation was active",
            duty_change,
        ),
        (
            axes[2],
            false_triggers,
            "Healthy False Triggers / min",
            "Stimulation epochs during healthy runs",
            ftr_change,
        ),
    ]

    for ax, values, title, ylabel, pct_change in panels:
        x = np.arange(len(labels), dtype=np.float64)
        bars = ax.bar(x, values, color=colors, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        ymax = float(max(values)) * 1.28 if np.isfinite(values).all() else 1.0
        ax.set_ylim(0.0, ymax)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ymax * 0.03,
                f"{value:.3f}" if value < 10 else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#222222",
            )

        direction = "lower" if pct_change < 0 else "higher"
        ax.text(
            0.5,
            0.92,
            f"HDC is {abs(pct_change):.1f}% {direction}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#1b1b1b",
            bbox={"facecolor": "#efe8d8", "edgecolor": "none", "boxstyle": "round,pad=0.35"},
        )

    fig.suptitle(
        "Presentation Summary: HDC vs Classical Beta aDBS",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    out = output_dir / "presentation_hdc_vs_beta_summary.png"
    return save_current(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/plots")
    args = parser.parse_args()
    out = generate(Path(args.output_dir))
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()
