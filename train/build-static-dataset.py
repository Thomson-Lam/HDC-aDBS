"""Build static ODE trajectory dataset.

Usage:
    uv run python train/build-static-dataset.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_static_dataset import BuildDatasetConfig, build_static_dataset


def main() -> None:
    cfg = BuildDatasetConfig()
    out_dir = build_static_dataset(cfg)
    print(f"Built static dataset: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    print(f"Build config: {out_dir / 'build_config.yaml'}")


if __name__ == "__main__":
    main()
