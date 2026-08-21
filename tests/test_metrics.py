from battery_lab.metrics import mae, rmse


def test_metrics_zero():
    assert rmse([1, 2], [1, 2]) == 0
    assert mae([1, 2], [1, 2]) == 0
