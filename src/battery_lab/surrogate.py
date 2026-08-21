from __future__ import annotations

import numpy as np


class RidgeSurrogate:
    """Small dependency-light ridge regressor for surrogate pipeline baselines."""

    def __init__(self, alpha: float = 1e-6) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    def fit(self, x, y) -> RidgeSurrogate:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2-D matrix")
        ones = np.ones((x_arr.shape[0], 1))
        design = np.hstack([ones, x_arr])
        reg = np.eye(design.shape[1]) * self.alpha
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + reg, design.T @ y_arr)
        return self

    def predict(self, x) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model has not been fitted")
        x_arr = np.asarray(x, dtype=float)
        design = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
        return design @ self.coef_
