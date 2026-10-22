import numpy as np
import pytest

from battery_lab.optimization import pareto_mask


def test_pareto_mask():
    values = np.array([[1, 3], [2, 2], [3, 1], [3, 3]], dtype=float)
    mask = pareto_mask(values)
    assert mask.tolist() == [True, True, True, False]


def test_pareto_mask_rejects_non_matrices():
    with pytest.raises(ValueError, match="2-D"):
        pareto_mask([1.0, 2.0])


def test_pareto_mask_rejects_nonfinite_objectives():
    with pytest.raises(ValueError):
        pareto_mask(np.array([[1.0, np.nan]]))
    assert pareto_mask(np.empty((0, 2))).tolist() == []
