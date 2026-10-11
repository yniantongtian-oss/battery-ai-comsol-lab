import numpy as np
import pandas as pd
import pytest

from battery_lab.data_io import validate_timeseries
from battery_lab.ecm import TheveninParameters, TheveninState, coulomb_count, thevenin_step
from battery_lab.metrics import mae, max_abs_error, rmse


@pytest.mark.parametrize("column", ["current_A", "temperature_K"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "invalid"])
def test_required_numeric_columns_reject_invalid_data(column, value):
    frame = pd.DataFrame(
        {
            "time_s": [0, 1],
            "current_A": [1, 1],
            "voltage_V": [3.7, 3.6],
            "temperature_K": [298, 299],
        }
    )
    frame[column] = [value, value]
    assert any(column in error for error in validate_timeseries(frame))


@pytest.mark.parametrize("metric", [rmse, mae, max_abs_error])
@pytest.mark.parametrize("invalid", [[], [np.nan], [np.inf]])
def test_metrics_reject_empty_or_nonfinite_arrays(metric, invalid):
    with pytest.raises(ValueError):
        metric(invalid, invalid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dt_s", -1),
        ("dt_s", np.nan),
        ("capacity_Ah", np.nan),
        ("capacity_Ah", np.inf),
        ("soc", np.nan),
        ("current_A", np.inf),
    ],
)
def test_coulomb_count_rejects_invalid_inputs(field, value):
    values = dict(soc=0.5, current_A=1.0, dt_s=1.0, capacity_Ah=2.0)
    values[field] = value
    with pytest.raises(ValueError):
        coulomb_count(**values)


def test_two_negative_rc_parameters_are_not_a_valid_time_constant():
    with pytest.raises(ValueError):
        thevenin_step(TheveninState(0.5), 1, 1, TheveninParameters(2, 0.01, -1, -10), 3.7)


def test_zero_time_step_and_charging_current_remain_supported():
    assert coulomb_count(0.5, -1, 0, 2) == 0.5
    assert coulomb_count(0.5, -1, 1, 2) > 0.5
