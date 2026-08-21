from __future__ import annotations

import numpy as np


def pareto_mask(objectives, minimize: bool = True) -> np.ndarray:
    """Return a boolean mask selecting non-dominated rows."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("objectives must be a 2-D array")
    data = values if minimize else -values
    keep = np.ones(data.shape[0], dtype=bool)
    for i, point in enumerate(data):
        if not keep[i]:
            continue
        dominated_by_other = np.any(
            np.all(data <= point, axis=1) & np.any(data < point, axis=1)
        )
        if dominated_by_other:
            keep[i] = False
    return keep
