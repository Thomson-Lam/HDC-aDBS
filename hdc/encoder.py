"""Window-to-hypervector encoder pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dictionaries import DictionaryConfig, build_dictionaries
from .primitives import bind, bundle


@dataclass(frozen=True)
class EncoderConfig:
    """HDC window encoder constructor config"""

    dimension: int
    n_bins: int
    window_length: int = 128
    value_init: str = "random"
    clip_z: float = 3.0
    seed: int | None = None
    normalization: str = "window_zscore"
    dataset_mean: float = 0.0
    dataset_std: float = 1.0


# the encoder designed for fixed length 1D signal windows
# this class encodes simulator output (ODE/LFP trajectories) into hypervectors for
# classif.
class WindowEncoder:
    """Encode fixed-length signal windows into bipolar hypervectors."""

    def __init__(self, config: EncoderConfig) -> None:
        if config.dimension <= 0:
            raise ValueError("dimension must be positive")
        if config.n_bins <= 1:
            raise ValueError("n_bins must be greater than 1")
        if config.window_length <= 0:
            raise ValueError("window_length must be positive")
        if config.clip_z <= 0:
            raise ValueError("clip_z must be positive")
        if config.normalization not in {"window_zscore", "dataset_zscore", "none"}:
            raise ValueError(
                "normalization must be one of {'window_zscore','dataset_zscore','none'}"
            )

        self.config = config
        dict_cfg = DictionaryConfig(
            dimension=config.dimension,
            n_bins=config.n_bins,
            window_length=config.window_length,
            value_init=config.value_init,
            seed=config.seed,
        )
        self.value_dict, self.position_dict = build_dictionaries(dict_cfg)

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @property
    def window_length(self) -> int:
        return self.config.window_length

    def normalize_window(self, window: np.ndarray) -> np.ndarray:
        """Normalize one window before quantization.

        Modes:
        - ``window_zscore``: normalize each window independently (legacy default)
        - ``dataset_zscore``: normalize with fixed dataset-level mean/std
        - ``none``: no normalization
        """
        x = np.asarray(window, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("window must be rank-1")
        if x.shape[0] != self.window_length:
            raise ValueError(
                f"window length must be {self.window_length}, got {x.shape[0]}"
            )

        mode = self.config.normalization
        if mode == "window_zscore":
            mean = np.mean(x)
            std = np.std(x)
            if std <= 0.0:
                return np.zeros_like(x)
            return (x - mean) / std
        if mode == "dataset_zscore":
            std = float(self.config.dataset_std)
            if std <= 0.0:
                raise ValueError("dataset_std must be > 0 for dataset_zscore")
            return (x - float(self.config.dataset_mean)) / std
        if mode == "none":
            return x
        raise ValueError(
            "normalization must be one of {'window_zscore','dataset_zscore','none'}"
        )

    # quantize is needed because encoder val dict is discrete but window is continuous
    # we convert each z-scored sample into a bin index so we can get value_dict  from raw float vals
    def quantize(self, z_window: np.ndarray) -> np.ndarray:
        """Quantize z-scored samples to bin indices in [0, n_bins-1]."""
        z = np.asarray(z_window, dtype=np.float64)
        clip = self.config.clip_z
        z = np.clip(z, -clip, clip)

        n_bins = self.config.n_bins
        scaled = (z + clip) / (2.0 * clip)
        bins = np.floor(scaled * n_bins).astype(np.int64)
        bins = np.clip(bins, 0, n_bins - 1)
        return bins

    def encode_window(self, window: np.ndarray) -> np.ndarray:
        """Encode one signal window into one bipolar hypervector."""
        z = self.normalize_window(window)
        bin_ids = self.quantize(z)

        value_hv = self.value_dict[bin_ids]
        bound = bind(value_hv, self.position_dict)
        encoded = bundle(bound, axis=0)
        return encoded.astype(np.int8, copy=False)

    def encode_batch(self, windows: np.ndarray) -> np.ndarray:
        """Encode a batch of windows with shape (n_windows, window_length)."""
        x = np.asarray(windows, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("windows must be rank-2")
        if x.shape[1] != self.window_length:
            raise ValueError(
                f"windows second dimension must be {self.window_length}, got {x.shape[1]}"
            )
        encoded = np.empty((x.shape[0], self.dimension), dtype=np.int8)
        for i in range(x.shape[0]):
            encoded[i] = self.encode_window(x[i])
        return encoded
