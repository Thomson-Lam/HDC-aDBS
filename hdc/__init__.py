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
]
