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
    df = read_csv_if_exists(ARTIFACTS / "encoder_search" / "leaderboard.csv")
    if df is None or df.empty:
        print("[skip] leaderboard.csv not found")
        return None

    set_plot_style()
    fig, ax = plt.subplots()
    for readout, color in (("prototype", "#2a9d8f"), ("linear", "#e76f51")):
        sub = df[df["readout"] == readout]
        if sub.empty:
            continue
        ax.scatter(
            sub["encoding_ms_per_window"],
            sub["balanced_accuracy_clean"],
            label=readout,
            c=color,
            s=55,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
        )

    ax.set_title("Encoder Search: Speed vs Clean Balanced Accuracy")
    ax.set_xlabel("Encoding Time (ms / window)")
    ax.set_ylabel("Validation Clean Balanced Accuracy")
    ax.legend(loc="best")
    out = output_dir / "search_speed_vs_accuracy.png"
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
