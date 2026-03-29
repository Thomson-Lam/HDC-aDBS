from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plots.common import ARTIFACTS, read_yaml_if_exists, save_current, set_plot_style


def generate(output_dir: Path) -> Path | None:
    report = read_yaml_if_exists(ARTIFACTS / "models" / "train_report.yaml")
    if report is None:
        print("[skip] train_report.yaml not found")
        return None

    labels = ["clean", "onset", "recovery", "moderate", "test"]
    proto = [
        report["prototype"]["balanced_accuracy"],
        report["prototype_onset"]["balanced_accuracy"],
        report["prototype_recovery"]["balanced_accuracy"],
        report["prototype_moderate"]["balanced_accuracy"],
        report["prototype_test"]["balanced_accuracy"],
    ]
    linear = [
        report["linear"]["balanced_accuracy"],
        report["linear_onset"]["balanced_accuracy"],
        report["linear_recovery"]["balanced_accuracy"],
        report["linear_moderate"]["balanced_accuracy"],
        report["linear_test"]["balanced_accuracy"],
    ]

    set_plot_style()
    fig, ax = plt.subplots()
    x = np.arange(len(labels), dtype=np.float64)
    w = 0.38
    ax.bar(x - w / 2.0, proto, width=w, label="prototype", color="#2a9d8f")
    ax.bar(x + w / 2.0, linear, width=w, label="linear", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Offline Performance: Prototype vs Linear")
    ax.legend(loc="best")
    out = output_dir / "offline_balanced_accuracy_by_subset.png"
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
