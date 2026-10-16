"""Positive Ridge stacking with the exact member normalization convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class PositiveRidgeStacker:
    """Fit a non-negative Ridge blend after population-std normalization."""

    alpha: float = 0.01

    def fit(
        self,
        members: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "PositiveRidgeStacker":
        members = np.asarray(members, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if members.ndim != 2:
            raise ValueError("members must be a two-dimensional matrix")
        if members.shape[0] != target.size or target.shape != sample_weight.shape:
            raise ValueError("training arrays must align")
        if not (
            np.isfinite(members).all()
            and np.isfinite(target).all()
            and np.isfinite(sample_weight).all()
        ):
            raise ValueError("training arrays must be finite")

        scale = members.std(axis=0, ddof=0)
        scale[scale < 1e-12] = 1.0
        model = Ridge(alpha=self.alpha, fit_intercept=False, positive=True)
        model.fit(members / scale, target, sample_weight=sample_weight)
        self.scale_ = scale
        self.coef_ = model.coef_ / scale
        return self

    def predict(self, members: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        members = np.asarray(members, dtype=np.float64)
        if members.ndim != 2 or members.shape[1] != self.coef_.size:
            raise ValueError("member matrix has an unexpected shape")
        return members @ self.coef_
