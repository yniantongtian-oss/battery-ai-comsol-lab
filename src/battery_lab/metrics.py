from __future__ import annotations

import numpy as np


def _validated_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("arrays must have the same shape")
    if not a.size:
        raise ValueError("arrays must not be empty")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("arrays must contain only finite values")
    return a, b


def rmse(y_true, y_pred) -> float:
    a, b = _validated_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    a, b = _validated_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def max_abs_error(y_true, y_pred) -> float:
    a, b = _validated_arrays(y_true, y_pred)
    return float(np.max(np.abs(a - b)))
