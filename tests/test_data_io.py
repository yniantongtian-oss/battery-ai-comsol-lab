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
