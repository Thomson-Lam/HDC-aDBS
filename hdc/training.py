"""
Training layer for HDC classifiers built on encoder primitives and readouts.
Contains training code that uses the model readbout defs and encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .encoder import EncoderConfig, WindowEncoder
from .readouts import LinearReadout, PrototypeReadout


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = np.unique(y_true)
    if labels.shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


class BaseHDCTrainer(ABC):
    """
    Base interface for window-level HDC trainers.

    Convention: label 0 = healthy, label 1 = pathological.
    """

    trainer_type: str

    def __init__(self, encoder_config: EncoderConfig) -> None:
        self.encoder_config = encoder_config
        self.encoder = WindowEncoder(encoder_config)
        self._is_fit = False

    def encode(self, windows: np.ndarray) -> np.ndarray:
        return self.encoder.encode_batch(windows)

    @abstractmethod
    def fit(self, windows: np.ndarray, y: np.ndarray) -> "BaseHDCTrainer":
        pass

    @abstractmethod
    def decision_function(self, windows: np.ndarray) -> np.ndarray:
        pass

    def predict(self, windows: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        scores = self.decision_function(windows)
        return (scores >= threshold).astype(np.int64)

    def evaluate(self, windows: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y = np.asarray(y).astype(np.int64)
        scores = self.decision_function(windows)
        pred = (scores >= 0.0).astype(np.int64)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auroc": _safe_auroc(y, scores),
        }

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseHDCTrainer":
        pass

    def _save_common(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "trainer_type": self.trainer_type,
            "encoder_config": asdict(self.encoder_config),
            "label_convention": {"healthy": 0, "pathological": 1},
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        np.savez_compressed(
            path / "encoder_dictionaries.npz",
            value_dict=self.encoder.value_dict,
            position_dict=self.encoder.position_dict,
        )

    @staticmethod
    def _load_common(path: Path) -> tuple[dict, EncoderConfig, np.ndarray, np.ndarray]:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        encoder_config = EncoderConfig(**metadata["encoder_config"])
        dictionaries = np.load(path / "encoder_dictionaries.npz")
        value_dict = dictionaries["value_dict"].astype(np.int8, copy=False)
        position_dict = dictionaries["position_dict"].astype(np.int8, copy=False)
        return metadata, encoder_config, value_dict, position_dict


class PrototypeHDCTrainer(BaseHDCTrainer):
    """Prototype-similarity HDC trainer."""

    trainer_type = "prototype"

    def __init__(self, encoder_config: EncoderConfig) -> None:
        super().__init__(encoder_config=encoder_config)
        self.readout = PrototypeReadout()

    def fit(self, windows: np.ndarray, y: np.ndarray) -> "PrototypeHDCTrainer":
        x = self.encode(windows)
        self.readout.fit(x, np.asarray(y).astype(np.int64))
        self._is_fit = True
        return self

    def decision_function(self, windows: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("trainer must be fit before inference")
        x = self.encode(windows)
        return self.readout.decision_function(x)

    def save(self, path: str | Path) -> None:
        path = _to_path(path)
        if not self._is_fit:
            raise RuntimeError("cannot save an unfitted trainer")
        self._save_common(path)
        np.savez_compressed(
            path / "readout.npz",
            healthy_prototype=self.readout.healthy_prototype,
            path_prototype=self.readout.path_prototype,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeHDCTrainer":
        path = _to_path(path)
        metadata, encoder_config, value_dict, position_dict = cls._load_common(path)
        if metadata["trainer_type"] != "prototype":
            raise ValueError("artifact trainer_type is not 'prototype'")

        trainer = cls(encoder_config=encoder_config)
        trainer.encoder.value_dict = value_dict
        trainer.encoder.position_dict = position_dict

        readout_data = np.load(path / "readout.npz")
        trainer.readout.healthy_prototype = readout_data["healthy_prototype"].astype(
            np.int8, copy=False
        )
        trainer.readout.path_prototype = readout_data["path_prototype"].astype(
            np.int8, copy=False
        )
        trainer._is_fit = True
        return trainer


class LinearHDCTrainer(BaseHDCTrainer):
    """Linear-classifier-over-HV HDC trainer."""

    trainer_type = "linear"

    def __init__(
        self,
        encoder_config: EncoderConfig,
        seed: int | None = None,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(encoder_config=encoder_config)
        self.seed = seed
        self.max_iter = max_iter
        self.readout = LinearReadout(seed=seed, max_iter=max_iter)

    def fit(self, windows: np.ndarray, y: np.ndarray) -> "LinearHDCTrainer":
        x = self.encode(windows)
        self.readout.fit(x, np.asarray(y).astype(np.int64))
        self._is_fit = True
        return self

    def decision_function(self, windows: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("trainer must be fit before inference")
        x = self.encode(windows)
        return self.readout.decision_function(x)

    def save(self, path: str | Path) -> None:
        path = _to_path(path)
        if not self._is_fit:
            raise RuntimeError("cannot save an unfitted trainer")
        self._save_common(path)

        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        metadata["linear_readout"] = {
            "seed": self.seed,
            "max_iter": self.max_iter,
            "solver": "liblinear",
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        np.savez_compressed(
            path / "readout.npz",
            coef=self.readout.model.coef_,
            intercept=self.readout.model.intercept_,
            classes=self.readout.model.classes_,
            n_features_in=np.array([self.readout.model.n_features_in_], dtype=np.int64),
            n_iter=self.readout.model.n_iter_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearHDCTrainer":
        path = _to_path(path)
        metadata, encoder_config, value_dict, position_dict = cls._load_common(path)
        if metadata["trainer_type"] != "linear":
            raise ValueError("artifact trainer_type is not 'linear'")

        linear_meta = metadata.get("linear_readout", {})
        trainer = cls(
            encoder_config=encoder_config,
            seed=linear_meta.get("seed"),
            max_iter=int(linear_meta.get("max_iter", 1000)),
        )
        trainer.encoder.value_dict = value_dict
        trainer.encoder.position_dict = position_dict

        readout_data = np.load(path / "readout.npz")
        model = trainer.readout.model
        model.coef_ = readout_data["coef"].astype(np.float64, copy=False)
        model.intercept_ = readout_data["intercept"].astype(np.float64, copy=False)
        model.classes_ = readout_data["classes"].astype(np.int64, copy=False)
        model.n_features_in_ = int(readout_data["n_features_in"][0])
        model.n_iter_ = readout_data["n_iter"].astype(np.int32, copy=False)

        trainer._is_fit = True
        return trainer
