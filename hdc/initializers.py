"""

Base Hypervector initialization definitions for HDC encoders.

This module provides:
- random binary initialization (classical HDC baseline)
- RFF-inspired correlated binary initialization from a target similarity matrix

All initializers return bipolar binary hypervectors in {-1, +1} with shape
(n_entities, dimension).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseHypervectorInitializer(ABC):
    """Abstract base class for hypervector dictionary initializers."""

    def __init__(self, dimension: int, seed: int | None = None) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.seed = seed

    @abstractmethod
    def initialize(self, n_entities: int) -> np.ndarray:
        """Return an array with shape (n_entities, dimension)."""

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


class RandomBinaryInitializer(BaseHypervectorInitializer):
    """Classical random binary initializer.

    Each hypervector entry is sampled i.i.d. from {-1, +1}.
    """

    def initialize(self, n_entities: int) -> np.ndarray:
        if n_entities <= 0:
            raise ValueError("n_entities must be positive")
        rng = self._rng()
        hv = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(n_entities, self.dimension)
        )
        return hv.astype(np.int8, copy=False)


class RFFBinaryInitializer(BaseHypervectorInitializer):
    """RFF-style correlated binary initializer for value/bin hypervectors.

    Assumptions used for the MVP testbed:
    - entities are quantization bins on a normalized line in [0, 1]
    - target similarity matrix uses an RBF kernel
    - sigma is fixed to bin spacing delta = 1 / (n_bins - 1)
    - binary output is bipolar {-1, +1}
    - negative eigenvalues from numerical noise are clipped to 0
    - sign ties map to +1
    """

    def __init__(
        self,
        dimension: int,
        seed: int | None = None,
        eigenvalue_floor: float = 0.0,
    ) -> None:
        super().__init__(dimension=dimension, seed=seed)
        if eigenvalue_floor < 0:
            raise ValueError("eigenvalue_floor must be >= 0")
        self.eigenvalue_floor = eigenvalue_floor

    @staticmethod
    def bin_centers(n_entities: int) -> np.ndarray:
        if n_entities <= 1:
            return np.array([0.0], dtype=np.float64)
        return np.linspace(0.0, 1.0, num=n_entities, dtype=np.float64)

    @staticmethod
    def sigma_from_bin_spacing(n_entities: int) -> float:
        if n_entities <= 1:
            return 1.0
        return 1.0 / float(n_entities - 1)

    def build_similarity_matrix(self, n_entities: int) -> np.ndarray:
        if n_entities <= 0:
            raise ValueError("n_entities must be positive")

        centers = self.bin_centers(n_entities)
        sigma = self.sigma_from_bin_spacing(n_entities)
        diff = centers[:, None] - centers[None, :]
        m = np.exp(-(diff**2) / (2.0 * sigma**2))

        # Numerical safety and explicit unit diagonal.
        m = 0.5 * (m + m.T)
        np.fill_diagonal(m, 1.0)
        return m

    @staticmethod
    def similarity_to_gaussian_covariance(similarity_matrix: np.ndarray) -> np.ndarray:
        return np.sin(0.5 * np.pi * similarity_matrix)

    def _project_to_psd(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = 0.5 * (matrix + matrix.T)
        evals, evecs = np.linalg.eigh(matrix)
        evals = np.maximum(evals, self.eigenvalue_floor)
        return evecs, evals

    def initialize(self, n_entities: int) -> np.ndarray:
        if n_entities <= 0:
            raise ValueError("n_entities must be positive")

        rng = self._rng()
        m = self.build_similarity_matrix(n_entities)
        sigma_hat = self.similarity_to_gaussian_covariance(m)
        u, lambdas = self._project_to_psd(sigma_hat)

        x = rng.standard_normal(size=(n_entities, self.dimension))
        z = u @ (np.sqrt(lambdas)[:, None] * x)

        hv = np.where(z >= 0.0, 1, -1).astype(np.int8)
        return hv

    @staticmethod
    def empirical_similarity(hypervectors: np.ndarray) -> np.ndarray:
        """Compute normalized inner-product similarity matrix."""
        if hypervectors.ndim != 2:
            raise ValueError("hypervectors must be rank-2")
        d = hypervectors.shape[1]
        if d == 0:
            raise ValueError("hypervectors must have nonzero dimension")
        hv = hypervectors.astype(np.float64)
        return (hv @ hv.T) / float(d)
