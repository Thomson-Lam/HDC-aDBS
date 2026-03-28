"""Window-to-hypervector encoder pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dictionaries import DictionaryConfig, build_dictionaries
from .primitives import bind, bundle


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the HDC window encoder."""

    dimension: int
    n_bins: int
    window_length: int = 128
    value_init: str = "random"
    clip_z: float = 3.0
    seed: int | None = None


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

    def zscore_window(self, window: np.ndarray) -> np.ndarray:
        """Apply per-window z-score normalization."""
        x = np.asarray(window, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("window must be rank-1")
        if x.shape[0] != self.window_length:
            raise ValueError(
                f"window length must be {self.window_length}, got {x.shape[0]}"
            )

        mean = np.mean(x)
        std = np.std(x)
        if std <= 0.0:
            return np.zeros_like(x)
        return (x - mean) / std

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
        z = self.zscore_window(window)
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
