"""
Readout model layer on top of encoded hypervectors for either using the raw
hypervectors directly for comparison or building a linear classifier on top of it during training. This would be the 'model' layer equivalent for ML but for HDC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression

from .primitives import bundle, normalized_dot


def _validate_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2:
        raise ValueError("X must be rank-2")
    if y.ndim != 1:
        raise ValueError("y must be rank-1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching first dimension")
    labels = np.unique(y)
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("labels must be binary with values in {0, 1}")
    return x, y


class BaseReadout(ABC):
    """
    Shared interface for HDC readout models.

    Convention: label 0 = healthy, label 1 = pathological/abnormal.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseReadout":
        pass

    @abstractmethod
    def decision_function(self, x: np.ndarray) -> np.ndarray:
        pass

    def predict(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        scores = self.decision_function(x)
        return (scores >= threshold).astype(np.int64)


class PrototypeReadout(BaseReadout):
    """
    Prototype similarity readout using margin scoring, uses hypervectors directly.
    """

    def __init__(self) -> None:
        self.healthy_prototype: np.ndarray | None = None
        self.path_prototype: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PrototypeReadout":
        x, y = _validate_xy(x, y)

        if not np.any(y == 0):
            raise ValueError("healthy class (0) is missing")
        if not np.any(y == 1):
            raise ValueError("pathological class (1) is missing")

        healthy = x[y == 0]
        pathological = x[y == 1]

        self.healthy_prototype = bundle(healthy, axis=0)
        self.path_prototype = bundle(pathological, axis=0)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.healthy_prototype is None or self.path_prototype is None:
            raise RuntimeError("readout must be fit before calling decision_function")
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        sim_path = normalized_dot(x, self.path_prototype)
        sim_healthy = normalized_dot(x, self.healthy_prototype)
        return sim_path - sim_healthy


class LinearReadout(BaseReadout):
    """Logistic Regression trained over encoded hypervectors."""

    def __init__(self, seed: int | None = None, max_iter: int = 1000) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.model = LogisticRegression(
            random_state=seed,
            max_iter=max_iter,
            solver="liblinear",
        )
    
    # fitting method for logreg.
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearReadout":
        x, y = _validate_xy(x, y)
        self.model.fit(x.astype(np.float64), y)
        return self

    # returns the probability score for classif from the model's raw linear score (margin)
    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        return self.model.decision_function(x.astype(np.float64))
