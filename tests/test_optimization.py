import numpy as np

from battery_lab.optimization import pareto_mask


def test_pareto_mask():
    values = np.array([[1, 3], [2, 2], [3, 1], [3, 3]], dtype=float)
    mask = pareto_mask(values)
    assert mask.tolist() == [True, True, True, False]
