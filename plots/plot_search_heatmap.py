from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plots.common import ARTIFACTS, read_csv_if_exists, save_current, set_plot_style


def generate(output_dir: Path) -> Path | None:
    df = read_csv_if_exists(ARTIFACTS / "encoder_search" / "leaderboard.csv")
    if df is None or df.empty:
        print("[skip] leaderboard.csv not found")
        return None

    pivot = (
        df.groupby(["dimension", "n_bins"], as_index=False)["balanced_accuracy_clean"]
        .max()
        .pivot(index="dimension", columns="n_bins", values="balanced_accuracy_clean")
        .sort_index()
    )
    if pivot.empty:
        print("[skip] no data for heatmap")
        return None

    set_plot_style()
    fig, ax = plt.subplots()
    arr = pivot.to_numpy(dtype=np.float64)
    im = ax.imshow(arr, cmap="magma", aspect="auto")
    ax.set_title("Encoder Search Heatmap: Best Clean Balanced Accuracy")
    ax.set_xlabel("n_bins")
    ax.set_ylabel("dimension")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            ax.text(c, r, f"{arr[r, c]:.3f}", ha="center", va="center", color="white")

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Balanced Accuracy")
    out = output_dir / "search_heatmap_balanced_accuracy.png"
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
