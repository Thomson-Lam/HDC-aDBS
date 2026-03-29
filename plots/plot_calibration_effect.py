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

    labels = ["prototype", "linear"]
    before = [
        float(report["prototype_holdout_false_trigger_rate"]),
        float(report["linear_holdout_false_trigger_rate"]),
    ]
    after = [
        float(report["prototype_calibrated"]["holdout_false_trigger_rate"]),
        float(report["linear_calibrated"]["holdout_false_trigger_rate"]),
    ]

    set_plot_style()
    fig, ax = plt.subplots()
    x = np.arange(len(labels), dtype=np.float64)
    w = 0.38
    ax.bar(x - w / 2.0, before, width=w, label="uncalibrated", color="#7b9acc")
    ax.bar(x + w / 2.0, after, width=w, label="calibrated", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Healthy Holdout False Trigger Rate")
    ax.set_title("Calibration Effect on Holdout False Triggers")
    ax.legend(loc="best")
    out = output_dir / "calibration_holdout_false_trigger_effect.png"
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
