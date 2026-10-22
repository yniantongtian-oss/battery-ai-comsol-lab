import pandas as pd

from battery_lab.data_io import validate_timeseries


def test_valid_timeseries():
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "current_A": [1.0, 1.0],
            "voltage_V": [4.1, 4.0],
            "temperature_K": [298.15, 298.2],
        }
    )
    assert validate_timeseries(frame) == []


def test_timeseries_rejects_decreasing_time_and_impossible_voltage():
    decreasing = pd.DataFrame(
        {
            "time_s": [1.0, 0.0],
            "current_A": [1.0, 1.0],
            "voltage_V": [4.0, 3.9],
            "temperature_K": [298.15, 298.15],
        }
    )
    assert any("time_s" in error for error in validate_timeseries(decreasing))
    hot = decreasing.copy()
    hot["time_s"] = [0.0, 1.0]
    hot["voltage_V"] = [4.0, 12.0]
    assert any("voltage_V" in error for error in validate_timeseries(hot))
