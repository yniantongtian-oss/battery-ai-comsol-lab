from __future__ import annotations

import numpy as np


def pareto_mask(objectives, minimize: bool = True) -> np.ndarray:
    """Return a boolean mask selecting non-dominated rows."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("objectives must be a 2-D array")
    if values.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if not np.all(np.isfinite(values)):
        raise ValueError("objectives must contain only finite values")
    data = values if minimize else -values
    keep = np.ones(data.shape[0], dtype=bool)
    for index, point in enumerate(data):
        dominated = np.all(data <= point, axis=1) & np.any(data < point, axis=1)
        dominated[index] = False
        if np.any(dominated):
            keep[index] = False
    return keep
