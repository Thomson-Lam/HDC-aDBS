"""HDC core utilities for encoding and readout."""

from .initializers import (
    BaseHypervectorInitializer,
    RandomBinaryInitializer,
    RFFBinaryInitializer,
)
from .primitives import bind, bipolarize, bundle, l2_normalize, normalized_dot
from .dictionaries import (
    DictionaryConfig,
    build_dictionaries,
    build_position_dictionary,
    build_value_dictionary,
)
from .encoder import EncoderConfig, WindowEncoder
from .readouts import BaseReadout, LinearReadout, PrototypeReadout
from .search import (
    EncoderSearchSpec,
    SearchConfig,
    SearchResult,
    SplitData,
    ValidationData,
    rank_results,
    run_validator_search,
    top_candidates,
    write_results_csv,
    write_results_jsonl,
)
from .training import BaseHDCTrainer, LinearHDCTrainer, PrototypeHDCTrainer

__all__ = [
    "BaseHypervectorInitializer",
    "RandomBinaryInitializer",
    "RFFBinaryInitializer",
    "bind",
    "bipolarize",
    "bundle",
    "l2_normalize",
    "normalized_dot",
    "DictionaryConfig",
    "build_dictionaries",
    "build_position_dictionary",
    "build_value_dictionary",
    "EncoderConfig",
    "WindowEncoder",
    "BaseReadout",
    "PrototypeReadout",
    "LinearReadout",
    "EncoderSearchSpec",
    "SearchConfig",
    "SplitData",
    "ValidationData",
    "SearchResult",
    "run_validator_search",
    "rank_results",
    "top_candidates",
    "write_results_jsonl",
    "write_results_csv",
    "BaseHDCTrainer",
    "PrototypeHDCTrainer",
    "LinearHDCTrainer",
]
