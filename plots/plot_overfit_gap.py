from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plots.common import ARTIFACTS, read_csv_if_exists, save_current, set_plot_style


def generate(output_dir: Path) -> Path | None:
    df = read_csv_if_exists(
        ARTIFACTS / "overfit_calibration" / "phase1_4_experiments.csv"
    )
    if df is None or df.empty:
        print("[skip] phase1_4_experiments.csv not found")
        return None

    df = df[df["normalization"] != "none"].copy()
    df = df[df["calibration_policy"] == "fpr_constrained"].copy()
    if df.empty:
        print("[skip] no filtered rows for overfit gap")
        return None

    set_plot_style()
    fig, ax = plt.subplots()
    for readout, color in (("prototype", "#2a9d8f"), ("linear", "#e76f51")):
        sub = df[df["readout"] == readout]
        if sub.empty:
            continue
        ax.scatter(
            sub["generalization_gap_auroc"],
            sub["val_clean_balanced_accuracy"],
            c=color,
            label=readout,
            s=52,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.25,
        )

    ax.set_title("Generalization Gap vs Validation Accuracy")
    ax.set_xlabel("Train AUROC - Val AUROC")
    ax.set_ylabel("Validation Clean Balanced Accuracy")
    ax.legend(loc="best")
    out = output_dir / "overfit_generalization_gap_vs_valacc.png"
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
