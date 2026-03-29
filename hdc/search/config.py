"""Fixed search-space configuration for the MVP encoder validator."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class EncoderSearchSpec:
    """One encoder candidate from the fixed validator search space."""

    dimension: int
    n_bins: int
    value_init: str


@dataclass(frozen=True)
class SearchConfig:
    """Frozen validator settings from project docs."""

    sampling_rate_hz: int = 250
    window_length: int = 128
    stride_ms: int = 50
    dimensions: tuple[int, ...] = (1000, 5000, 10000)
    bins: tuple[int, ...] = (8, 16)
    value_initializers: tuple[str, ...] = ("random", "rff")
    readouts: tuple[str, ...] = ("prototype", "linear")
    clip_z: float = 3.0
    base_seed: int = 123
    min_guardrail_auroc: float = 0.60
    min_moderate_auroc: float = 0.55
    max_holdout_false_trigger_rate: float = 0.40
    top_k: int = 3
    linear_c_candidates: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    calibration_target_val_fpr: float = 0.20
    allow_guardrail_fallback: bool = True

    def iter_encoder_specs(self) -> list[EncoderSearchSpec]:
        """Return all frozen encoder candidates.

        For the locked MVP this should produce 12 candidates.
        """
        specs = [
            EncoderSearchSpec(dimension=d, n_bins=b, value_init=init)
            for d, b, init in product(
                self.dimensions, self.bins, self.value_initializers
            )
        ]
        return specs
