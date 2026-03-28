"""Deterministic hypervector dictionary builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .initializers import RandomBinaryInitializer, RFFBinaryInitializer


@dataclass(frozen=True)
class DictionaryConfig:
    """Configuration for value and position hypervector dictionaries."""

    dimension: int
    n_bins: int
    window_length: int
    value_init: str = "random"
    seed: int | None = None


# helper to make separate seeds from a base seed to prevent identical RNG streams for the same dictionary 
def _seed_offset(seed: int | None, offset: int) -> int | None:
    if seed is None:
        return None
    return int(seed) + int(offset)


def build_value_dictionary(
    n_bins: int,
    dimension: int,
    init: str = "random",
    seed: int | None = None,
) -> np.ndarray:
    """Build value/bin hypervector dictionary with requested initializer."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if init == "random":
        initializer = RandomBinaryInitializer(dimension=dimension, seed=seed)
    elif init == "rff":
        initializer = RFFBinaryInitializer(dimension=dimension, seed=seed)
    else:
        raise ValueError("init must be one of {'random', 'rff'}")
    return initializer.initialize(n_bins)


def build_position_dictionary(
    window_length: int,
    dimension: int,
    seed: int | None = None,
) -> np.ndarray:
    """Build position hypervector dictionary.

    Position vectors are fixed to random binary for the MVP.
    """
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    initializer = RandomBinaryInitializer(dimension=dimension, seed=seed)
    return initializer.initialize(window_length)

# build the dictionaries with different seeds for the hypervectors.
def build_dictionaries(config: DictionaryConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build value and position dictionaries with deterministic seed offsets."""
    value_seed = _seed_offset(config.seed, 0)
    position_seed = _seed_offset(config.seed, 1)
    value_dict = build_value_dictionary(
        n_bins=config.n_bins,
        dimension=config.dimension,
        init=config.value_init,
        seed=value_seed,
    )
    position_dict = build_position_dictionary(
        window_length=config.window_length,
        dimension=config.dimension,
        seed=position_seed,
    )
    return value_dict, position_dict
