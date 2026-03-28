"""Core hypervector primitives for bipolar binary HDC.

All binary hypervectors use the bipolar convention {-1, +1}.
"""

from __future__ import annotations

import numpy as np


def bipolarize(x: np.ndarray, zero_to: int = 1) -> np.ndarray:
    """Convert an array to bipolar {-1, +1} values.

    Args:
        x: Input array.
        zero_to: Tie value used when entries equal zero, either +1 or -1.
    """
    if zero_to not in (-1, 1):
        raise ValueError("zero_to must be either -1 or +1")
    out = np.where(x > 0, 1, -1)
    out = np.where(x == 0, zero_to, out)
    return out.astype(np.int8, copy=False)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two bipolar hypervectors with elementwise sign-product."""
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return (a.astype(np.int8, copy=False) * b.astype(np.int8, copy=False)).astype(
        np.int8, copy=False
    )


def bundle(hypervectors: np.ndarray, axis: int = 0, zero_to: int = 1) -> np.ndarray:
    """Bundle hypervectors via majority vote then bipolarize.

    For bipolar vectors, bundling is sum followed by sign.
    """
    if hypervectors.size == 0:
        raise ValueError("hypervectors must be non-empty")
    summed = np.sum(hypervectors, axis=axis, dtype=np.int32)
    return bipolarize(summed, zero_to=zero_to)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Return L2-normalized array along axis."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def normalized_dot(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute cosine-like normalized dot similarity.

    Supports vector-vector and matrix-vector cases.
    """
    a2 = np.atleast_2d(a).astype(np.float64)
    b2 = np.atleast_2d(b).astype(np.float64)

    if b2.shape[0] != 1:
        raise ValueError("b must be a single vector or rank-1 array")
    if a2.shape[1] != b2.shape[1]:
        raise ValueError("a and b must share the same feature dimension")

    a_norm = l2_normalize(a2, axis=1, eps=eps)
    b_norm = l2_normalize(b2, axis=1, eps=eps)
    sims = a_norm @ b_norm.T
    sims = sims[:, 0]

    if np.ndim(a) == 1:
        return np.asarray(sims[0])
    return sims
