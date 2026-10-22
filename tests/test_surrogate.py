import numpy as np
import pandas as pd
import pytest

from battery_lab.metrics import rmse
from battery_lab.surrogate import (
    GreyBoxTheveninSurrogate,
    LumpedThermalSurrogate,
    ProtocolRidgeSurrogate,
    RidgeSurrogate,
    _best_thermal_pair,
    arrhenius_factor,
    coulomb_soc,
    euler_thermal_rollout,
    grey_box_voltage,
    ocv_derivative_min,
)


def test_ridge_surrogate_linear_fit():
    x = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2 * x[:, 0] + 1
    model = RidgeSurrogate(alpha=1e-9).fit(x, y)
    pred = model.predict([[3.0]])
    assert abs(float(pred[0]) - 7.0) < 1e-4


def test_ridge_rejects_bad_shapes_and_survives_rank_deficiency():
    with pytest.raises(ValueError):
        RidgeSurrogate(alpha=-1)
    model = RidgeSurrogate().fit([[0.0], [1.0]], [1.0, 3.0])
    with pytest.raises(RuntimeError):
        RidgeSurrogate().predict([[1.0]])
    with pytest.raises(ValueError):
        model.predict([[1.0, 2.0]])
    with pytest.raises(ValueError):
        RidgeSurrogate().fit([[1.0, 2.0]], [1.0, 2.0])
    underdetermined = RidgeSurrogate(alpha=1e-6).fit([[1.0, 2.0]], [3.0])
    assert np.all(np.isfinite(underdetermined.predict([[1.0, 2.0]])))


def test_grey_box_recovers_noiseless_parameters():
    capacity = 5.0
    soc_start = 1.0
    r0, r1, tau = 0.02, 0.03, 40.0
    coefficients = (2.8, 0.9, 0.0, 0.2)
    frames = []
    for c_rate, temperature in ((0.5, 298.15), (1.0, 313.15)):
        current = c_rate * capacity
        time = np.arange(0.0, 1200.0, 10.0)
        soc = coulomb_soc(time, current, capacity, soc_start)
        voltage = grey_box_voltage(
            time, current, temperature, soc, r0, r1, tau, coefficients
        )
        frames.append(
            pd.DataFrame(
                {
                    "time_s": time,
                    "current_A": np.full(time.shape, current),
                    "ambient_temperature_K": np.full(time.shape, temperature),
                    "voltage_V": voltage,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    model = GreyBoxTheveninSurrogate(capacity_Ah=capacity, soc_start=soc_start).fit(
        frame, tau_grid_s=np.array([20.0, 40.0, 60.0])
    )
    assert model.tau_s == 40.0
    assert model.r0_ohm == pytest.approx(r0, abs=1e-6)
    assert model.r1_ohm == pytest.approx(r1, abs=1e-6)
    assert model.ocv_coefficients == pytest.approx(coefficients, abs=1e-5)
    assert rmse(frame["voltage_V"], model.predict_voltage(frame)) < 1e-6
    assert model.resistances_nonnegative
    assert model.ocv_monotone_on_unit_interval


def test_surrogate_helpers_reject_unusable_inputs():
    with pytest.raises(ValueError):
        arrhenius_factor([-1.0])
    with pytest.raises(ValueError):
        coulomb_soc([0.0], [1.0], 0.0, 1.0)
    with pytest.raises(ValueError):
        grey_box_voltage([0.0], [1.0], [298.15], [1.0], 0.01, 0.01, 0.0, [3.0])
    with pytest.raises(ValueError):
        GreyBoxTheveninSurrogate(capacity_Ah=-1)
    with pytest.raises(ValueError):
        GreyBoxTheveninSurrogate(soc_start=2)
    with pytest.raises(RuntimeError):
        GreyBoxTheveninSurrogate().predict_voltage(pd.DataFrame({"time_s": [0.0]}))
    with pytest.raises(ValueError):
        GreyBoxTheveninSurrogate().fit(pd.DataFrame({"time_s": [0.0]}), [10.0])
    with pytest.raises(ValueError):
        LumpedThermalSurrogate().fit([], dt_s=1.0)
    with pytest.raises(RuntimeError):
        LumpedThermalSurrogate().predict_temperature([], dt_s=1.0)
    with pytest.raises(RuntimeError):
        _ = LumpedThermalSurrogate().thermal_capacity_J_per_K
    with pytest.raises(RuntimeError):
        _ = LumpedThermalSurrogate().heat_transfer_W_per_K
    with pytest.raises(RuntimeError):
        ProtocolRidgeSurrogate().predict(pd.DataFrame())
    with pytest.raises(ValueError):
        euler_thermal_rollout(298.15, [1.0, 1.0], 298.15, 0.0, 0.01, 0.001)
    assert euler_thermal_rollout(298.15, [], 298.15, 1.0, 0.01, 0.001).size == 0
    assert ocv_derivative_min([2.8, 0.2]) == pytest.approx(0.2)
    assert ocv_derivative_min([2.8, 0.4, -0.3, 0.2]) < 0.4
    with pytest.raises(ValueError):
        RidgeSurrogate(alpha=True)
    with pytest.raises(ValueError):
        RidgeSurrogate(alpha=float("nan"))
    with pytest.raises(ValueError):
        RidgeSurrogate().fit([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        RidgeSurrogate().fit([[1.0], [2.0]], [[[1.0]], [[2.0]]])
    with pytest.raises(ValueError):
        RidgeSurrogate().fit([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        RidgeSurrogate().fit(np.zeros((0, 1)), np.zeros((0,)))
    with pytest.raises(ValueError):
        RidgeSurrogate().fit([[float("nan")]], [1.0])
    fitted = RidgeSurrogate().fit([[1.0], [2.0]], [[3.0], [4.0]])
    with pytest.raises(ValueError):
        fitted.predict([[[1.0]]])
    with pytest.raises(ValueError):
        fitted.predict([[1.0, 2.0]])
    with pytest.raises(ValueError):
        fitted.predict([[float("nan")]])
    assert fitted.predict([1.0]).shape[0] == 1
    with pytest.raises(ValueError):
        GreyBoxTheveninSurrogate().fit(
            pd.DataFrame(
                {
                    "time_s": [0.0],
                    "current_A": [1.0],
                    "ambient_temperature_K": [298.15],
                    "voltage_V": [3.0],
                }
            ),
            [],
        )
    with pytest.raises(ValueError):
        LumpedThermalSurrogate().fit(
            [{"temperature_K": [1.0], "heat_W": [1.0, 2.0], "ambient_temperature_K": 298.15}],
            dt_s=1.0,
        )
    with pytest.raises(ValueError):
        LumpedThermalSurrogate().fit(
            [
                {
                    "temperature_K": [298.0, float("nan")],
                    "heat_W": [1.0, 1.0],
                    "ambient_temperature_K": 298.15,
                }
            ],
            dt_s=1.0,
        )
    with pytest.raises(ValueError):
        LumpedThermalSurrogate().fit(
            [
                {
                    "temperature_K": [298.0, 299.0],
                    "heat_W": [1.0, 1.0],
                    "ambient_temperature_K": float("nan"),
                }
            ],
            dt_s=1.0,
        )
    with pytest.raises(ValueError, match="dt_s"):
        LumpedThermalSurrogate().fit(
            [
                {
                    "temperature_K": [298.0, 299.0],
                    "heat_W": [1.0, 1.0],
                    "ambient_temperature_K": 298.15,
                }
            ],
            dt_s=0.0,
        )
    with pytest.raises(RuntimeError):
        _best_thermal_pair([], 1.0, [], [])
    ridge = ProtocolRidgeSurrogate()
    with pytest.raises(ValueError, match="missing columns"):
        ridge.fit(pd.DataFrame({"time_s": [0.0, 1.0]}), "voltage_V")
    with pytest.raises(ValueError, match="finite"):
        ridge.fit(
            pd.DataFrame(
                {
                    "time_s": [0.0, float("nan")],
                    "current_A": [1.0, 1.0],
                    "ambient_temperature_K": [298.15, 298.15],
                    "voltage_V": [3.0, 3.0],
                }
            ),
            "voltage_V",
        )
    electrical = pd.DataFrame(
        {
            "time_s": [],
            "current_A": [],
            "ambient_temperature_K": [],
            "voltage_V": [],
        }
    )
    with pytest.raises(ValueError, match="at least one row"):
        GreyBoxTheveninSurrogate().fit(electrical, [10.0])
    electrical = pd.DataFrame(
        {
            "time_s": [0.0],
            "current_A": [float("nan")],
            "ambient_temperature_K": [298.15],
            "voltage_V": [3.0],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        GreyBoxTheveninSurrogate().fit(electrical, [10.0])


def test_lumped_thermal_rollout_matches_a_known_trajectory():
    heat = np.full(40, 2.0)
    ambient = 298.15
    inverse_capacity = 1.0 / 80.0
    cooling = 0.004
    clean = euler_thermal_rollout(ambient, heat, ambient, 5.0, inverse_capacity, cooling)
    model = LumpedThermalSurrogate().fit(
        [{"temperature_K": clean, "heat_W": heat, "ambient_temperature_K": ambient}],
        dt_s=5.0,
    )
    predicted = model.predict_temperature(
        [{"temperature_K": clean, "heat_W": heat, "ambient_temperature_K": ambient}],
        dt_s=5.0,
    )
    assert rmse(clean, predicted) < 1e-2
    assert model.thermal_capacity_J_per_K > 0
    assert model.heat_transfer_W_per_K > 0
