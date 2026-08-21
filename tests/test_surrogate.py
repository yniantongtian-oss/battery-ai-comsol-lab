import numpy as np

from battery_lab.surrogate import RidgeSurrogate


def test_ridge_surrogate_linear_fit():
    x = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2 * x[:, 0] + 1
    model = RidgeSurrogate(alpha=1e-9).fit(x, y)
    pred = model.predict([[3.0]])
    assert abs(float(pred[0]) - 7.0) < 1e-4
