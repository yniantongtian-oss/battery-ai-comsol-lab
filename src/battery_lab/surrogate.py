"""Surrogate baselines for fast battery digital twins.

`RidgeSurrogate` is an unstructured linear baseline. `GreyBoxTheveninSurrogate`
identifies a cubic open-circuit-voltage curve and one RC pair whose resistances
follow a fixed Arrhenius prior. `LumpedThermalSurrogate` identifies the lumped
energy balance that goes with that electrical model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

GAS_CONSTANT_J_PER_MOL_K = 8.314462618
REFERENCE_TEMPERATURE_K = 298.15
ACTIVATION_ENERGY_J_PER_MOL = 25_000.0
NOMINAL_CAPACITY_AH = 5.0


def arrhenius_factor(temperature_K) -> np.ndarray:
    """Resistance scale relative to the reference temperature."""
    temperature = np.asarray(temperature_K, dtype=float)
    if np.any(temperature <= 0):
        raise ValueError("temperature_K must be positive")
    exponent = ACTIVATION_ENERGY_J_PER_MOL / GAS_CONSTANT_J_PER_MOL_K * (
        1.0 / temperature - 1.0 / REFERENCE_TEMPERATURE_K
    )
    return np.exp(exponent)


def ocv_polynomial(soc, coefficients) -> np.ndarray:
    state = np.asarray(soc, dtype=float)
    value = np.zeros(state.shape, dtype=float)
    for power, coefficient in enumerate(coefficients):
        value = value + float(coefficient) * state**power
    return value


def coulomb_soc(time_s, current_A, capacity_Ah: float, soc_start: float) -> np.ndarray:
    if capacity_Ah <= 0:
        raise ValueError("capacity_Ah must be positive")
    charge = np.asarray(current_A, dtype=float) * np.asarray(time_s, dtype=float)
    return float(soc_start) - charge / (capacity_Ah * 3600.0)


def grey_box_voltage(
    time_s,
    current_A,
    ambient_temperature_K,
    soc,
    r0_ohm: float,
    r1_ohm: float,
    tau_s: float,
    ocv_coefficients,
) -> np.ndarray:
    """Closed-form constant-current voltage from a rest state at t = 0."""
    if tau_s <= 0:
        raise ValueError("tau_s must be positive")
    arrhenius = arrhenius_factor(ambient_temperature_K)
    time = np.asarray(time_s, dtype=float)
    current = np.asarray(current_A, dtype=float)
    polarization = current * r1_ohm * arrhenius * (1.0 - np.exp(-time / tau_s))
    ohmic = current * r0_ohm * arrhenius
    return ocv_polynomial(soc, ocv_coefficients) - ohmic - polarization


def electrical_heat_W(
    time_s,
    current_A,
    ambient_temperature_K,
    r0_ohm: float,
    r1_ohm: float,
    tau_s: float,
):
    """Ohmic plus RC dissipation for the grey-box trajectory."""
    arrhenius = arrhenius_factor(ambient_temperature_K)
    current = np.asarray(current_A, dtype=float)
    r0_t = r0_ohm * arrhenius
    r1_t = np.maximum(r1_ohm * arrhenius, 1e-12)
    v_rc = current * r1_ohm * arrhenius * (1.0 - np.exp(-np.asarray(time_s, dtype=float) / tau_s))
    return current**2 * r0_t + v_rc**2 / r1_t


def euler_thermal_rollout(
    temperature0: float,
    heat_W,
    ambient_temperature_K: float,
    dt_s: float,
    inverse_capacity_K_per_J: float,
    cooling_rate_per_s: float,
) -> np.ndarray:
    """Explicit energy balance: C dT/dt = Q - hA (T - T_amb)."""
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    heat = np.asarray(heat_W, dtype=float)
    temperature = np.empty(heat.shape[0], dtype=float)
    if temperature.size == 0:
        return temperature
    temperature[0] = float(temperature0)
    ambient = float(ambient_temperature_K)
    for index in range(heat.shape[0] - 1):
        rise = inverse_capacity_K_per_J * heat[index]
        cooling = cooling_rate_per_s * (temperature[index] - ambient)
        temperature[index + 1] = temperature[index] + dt_s * (rise - cooling)
    return temperature


def ocv_derivative_min(coefficients) -> float:
    """Minimum dOCV/dSOC of a cubic on the unit interval."""
    coeffs = [float(value) for value in coefficients]
    while len(coeffs) < 4:
        coeffs.append(0.0)
    linear, quadratic, cubic = coeffs[1], coeffs[2], coeffs[3]
    candidates = [0.0, 1.0]
    if cubic != 0.0:
        vertex = -quadratic / (3.0 * cubic)
        if 0.0 < vertex < 1.0:
            candidates.append(vertex)
    slopes = [linear + 2.0 * quadratic * site + 3.0 * cubic * site**2 for site in candidates]
    return float(min(slopes))


class RidgeSurrogate:
    """Small dependency-light ridge regressor for surrogate pipeline baselines."""

    def __init__(self, alpha: float = 1e-6) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a finite non-negative number")
        if not math.isfinite(float(alpha)) or float(alpha) < 0:
            raise ValueError("alpha must be a finite non-negative number")
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(self, x, y) -> RidgeSurrogate:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2-D matrix")
        squeeze = False
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
            squeeze = True
        elif y_arr.ndim != 2:
            raise ValueError("y must be a 1-D or 2-D array")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if x_arr.shape[0] == 0:
            raise ValueError("x must contain at least one row")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("x and y must contain only finite values")
        design = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        coef, *_ = np.linalg.lstsq(gram, design.T @ y_arr, rcond=None)
        self.coef_ = coef[:, 0] if squeeze else coef
        self.n_features_ = int(x_arr.shape[1])
        return self

    def predict(self, x) -> np.ndarray:
        if self.coef_ is None or self.n_features_ is None:
            raise RuntimeError("model has not been fitted")
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2-D matrix")
        if x_arr.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {x_arr.shape[1]}")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x must contain only finite values")
        design = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
        return np.asarray(design @ self.coef_, dtype=float)


class GreyBoxTheveninSurrogate:
    """Identify OCV coefficients and nonnegative RC resistances.

    The activation energy and nominal capacity are fixed literature priors.
    Time is seconds from a rested state. SOC is coulomb-counted from `soc_start`.
    """

    def __init__(
        self,
        capacity_Ah: float = NOMINAL_CAPACITY_AH,
        soc_start: float = 1.0,
    ) -> None:
        if capacity_Ah <= 0:
            raise ValueError("capacity_Ah must be positive")
        if not 0.0 <= float(soc_start) <= 1.0:
            raise ValueError("soc_start must be between 0 and 1")
        self.capacity_Ah = float(capacity_Ah)
        self.soc_start = float(soc_start)
        self.r0_ohm: float | None = None
        self.r1_ohm: float | None = None
        self.tau_s: float | None = None
        self.ocv_coefficients: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, tau_grid_s) -> GreyBoxTheveninSurrogate:
        time_s, current_A, ambient_K, voltage = _electrical_columns(frame)
        soc = coulomb_soc(time_s, current_A, self.capacity_Ah, self.soc_start)
        grid = np.asarray(tau_grid_s, dtype=float).reshape(-1)
        if grid.size == 0 or np.any(grid <= 0) or not np.all(np.isfinite(grid)):
            raise ValueError("tau_grid_s must contain positive finite time constants")
        best_sse = math.inf
        best: tuple[float, np.ndarray] | None = None
        for tau in grid:
            design = _grey_box_design(time_s, current_A, ambient_K, soc, float(tau))
            coefficients = _fit_nonnegative_resistances(design, voltage)
            residual = voltage - design @ coefficients
            sse = float(residual @ residual)
            if sse < best_sse:
                best_sse = sse
                best = (float(tau), coefficients)
        if best is None:
            raise RuntimeError("grey-box fit did not produce a candidate")
        self.tau_s = best[0]
        self.ocv_coefficients = best[1][:4].copy()
        self.r0_ohm = float(best[1][4])
        self.r1_ohm = float(best[1][5])
        return self

    def predict_voltage(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fit()
        time_s, current_A, ambient_K, _ = _electrical_columns(frame, require_voltage=False)
        soc = coulomb_soc(time_s, current_A, self.capacity_Ah, self.soc_start)
        return grey_box_voltage(
            time_s,
            current_A,
            ambient_K,
            soc,
            float(self.r0_ohm),
            float(self.r1_ohm),
            float(self.tau_s),
            self.ocv_coefficients,
        )

    def predict_heat_W(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fit()
        time_s, current_A, ambient_K, _ = _electrical_columns(frame, require_voltage=False)
        return electrical_heat_W(
            time_s,
            current_A,
            ambient_K,
            float(self.r0_ohm),
            float(self.r1_ohm),
            float(self.tau_s),
        )

    @property
    def resistances_nonnegative(self) -> bool:
        self._require_fit()
        return float(self.r0_ohm) >= 0.0 and float(self.r1_ohm) >= 0.0

    @property
    def ocv_monotone_on_unit_interval(self) -> bool:
        self._require_fit()
        return ocv_derivative_min(self.ocv_coefficients) >= -1e-8

    def _require_fit(self) -> None:
        if self.ocv_coefficients is None or self.tau_s is None:
            raise RuntimeError("model has not been fitted")


class LumpedThermalSurrogate:
    """Grid-search the lumped thermal capacity and heat-transfer coefficient."""

    def __init__(self) -> None:
        self.inverse_capacity_K_per_J: float | None = None
        self.cooling_rate_per_s: float | None = None

    def fit(
        self, segments: list[dict[str, np.ndarray | float]], dt_s: float
    ) -> LumpedThermalSurrogate:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if not segments:
            raise ValueError("segments must not be empty")
        prepared = [_prepare_segment(segment) for segment in segments]
        capacity_axis = np.geomspace(1.0 / 400.0, 1.0 / 15.0, 16)
        cooling_axis = np.geomspace(1e-4, 5e-2, 16)
        best_capacity, best_cooling, _ = _best_thermal_pair(
            prepared, dt_s, capacity_axis, cooling_axis
        )
        capacity_zoom = np.geomspace(best_capacity / 1.6, best_capacity * 1.6, 13)
        cooling_zoom = np.geomspace(max(best_cooling / 1.6, 1e-6), best_cooling * 1.6, 13)
        best_capacity, best_cooling, _ = _best_thermal_pair(
            prepared, dt_s, capacity_zoom, cooling_zoom
        )
        self.inverse_capacity_K_per_J = float(best_capacity)
        self.cooling_rate_per_s = float(best_cooling)
        return self

    def predict_temperature(
        self, segments: list[dict[str, np.ndarray | float]], dt_s: float
    ) -> np.ndarray:
        if self.inverse_capacity_K_per_J is None or self.cooling_rate_per_s is None:
            raise RuntimeError("model has not been fitted")
        pieces = []
        for segment in segments:
            prepared = _prepare_segment(segment)
            pieces.append(
                euler_thermal_rollout(
                    prepared["temperature0"],
                    prepared["heat_W"],
                    prepared["ambient_temperature_K"],
                    dt_s,
                    self.inverse_capacity_K_per_J,
                    self.cooling_rate_per_s,
                )
            )
        return np.concatenate(pieces) if pieces else np.array([], dtype=float)

    @property
    def thermal_capacity_J_per_K(self) -> float:
        if not self.inverse_capacity_K_per_J:
            raise RuntimeError("model has not been fitted")
        return 1.0 / self.inverse_capacity_K_per_J

    @property
    def heat_transfer_W_per_K(self) -> float:
        if self.cooling_rate_per_s is None or not self.inverse_capacity_K_per_J:
            raise RuntimeError("model has not been fitted")
        return self.cooling_rate_per_s / self.inverse_capacity_K_per_J


class ProtocolRidgeSurrogate:
    """Standardized ridge on time, current, and ambient temperature."""

    _COLUMNS = ("time_s", "current_A", "ambient_temperature_K")

    def __init__(self, alpha: float = 1e-4) -> None:
        self.alpha = alpha
        self._model = RidgeSurrogate(alpha=alpha)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.target_name: str | None = None

    def fit(self, frame: pd.DataFrame, target: str) -> ProtocolRidgeSurrogate:
        features = _protocol_features(frame, self._COLUMNS)
        target_values = frame[target].to_numpy(dtype=float)
        self._mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self._scale = scale
        self._model.fit((features - self._mean) / self._scale, target_values)
        self.target_name = target
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._mean is None or self._scale is None:
            raise RuntimeError("model has not been fitted")
        features = _protocol_features(frame, self._COLUMNS)
        return self._model.predict((features - self._mean) / self._scale)


def _protocol_features(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("feature columns must be finite")
    return values


def _electrical_columns(frame: pd.DataFrame, require_voltage: bool = True):
    columns = ["time_s", "current_A", "ambient_temperature_K"]
    if require_voltage:
        columns.append("voltage_V")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    time_s = frame["time_s"].to_numpy(dtype=float)
    current_A = frame["current_A"].to_numpy(dtype=float)
    ambient_K = frame["ambient_temperature_K"].to_numpy(dtype=float)
    voltage = frame["voltage_V"].to_numpy(dtype=float) if require_voltage else None
    arrays = [time_s, current_A, ambient_K]
    if voltage is not None:
        arrays.append(voltage)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("electrical columns must be finite")
    if time_s.size == 0:
        raise ValueError("frame must contain at least one row")
    return time_s, current_A, ambient_K, voltage


def _grey_box_design(time_s, current_A, ambient_K, soc, tau_s: float) -> np.ndarray:
    arrhenius = arrhenius_factor(ambient_K)
    decay = 1.0 - np.exp(-time_s / tau_s)
    return np.column_stack(
        [
            np.ones(time_s.shape[0]),
            soc,
            soc**2,
            soc**3,
            -current_A * arrhenius,
            -current_A * arrhenius * decay,
        ]
    )


def _fit_nonnegative_resistances(design: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    """Least squares with R0 and R1 constrained to be nonnegative."""
    resistance_indexes = (4, 5)
    candidates = [set()]
    for index in resistance_indexes:
        candidates.extend([{index}, set(resistance_indexes)])
    best_sse = math.inf
    best = None
    seen: list[frozenset[int]] = []
    for fixed in candidates:
        key = frozenset(fixed)
        if key in seen:
            continue
        seen.append(key)
        coefficients = _solve_with_fixed(design, voltage, {index: 0.0 for index in fixed})
        if coefficients[4] < -1e-10 or coefficients[5] < -1e-10:
            continue
        coefficients[4] = max(float(coefficients[4]), 0.0)
        coefficients[5] = max(float(coefficients[5]), 0.0)
        residual = voltage - design @ coefficients
        sse = float(residual @ residual)
        if sse < best_sse:
            best_sse = sse
            best = coefficients
    if best is None:
        fallback = _solve_with_fixed(design, voltage, {4: 0.0, 5: 0.0})
        fallback[4] = 0.0
        fallback[5] = 0.0
        return fallback
    return best


def _solve_with_fixed(
    design: np.ndarray, voltage: np.ndarray, fixed: dict[int, float]
) -> np.ndarray:
    free = [index for index in range(design.shape[1]) if index not in fixed]
    adjusted = np.array(voltage, dtype=float, copy=True)
    for index, value in fixed.items():
        adjusted = adjusted - design[:, index] * value
    solved, *_ = np.linalg.lstsq(design[:, free], adjusted, rcond=None)
    coefficients = np.zeros(design.shape[1], dtype=float)
    for index, value in fixed.items():
        coefficients[index] = value
    for position, index in enumerate(free):
        coefficients[index] = float(solved[position])
    return coefficients


def _prepare_segment(segment: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    temperature = np.asarray(segment["temperature_K"], dtype=float)
    heat = np.asarray(segment["heat_W"], dtype=float)
    if temperature.shape != heat.shape or temperature.ndim != 1 or temperature.size < 2:
        raise ValueError("each thermal segment needs matching temperature and heat series")
    if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(heat)):
        raise ValueError("thermal segments must be finite")
    ambient = float(segment["ambient_temperature_K"])
    if not math.isfinite(ambient):
        raise ValueError("ambient_temperature_K must be finite")
    return {
        "temperature_K": temperature,
        "heat_W": heat,
        "ambient_temperature_K": ambient,
        "temperature0": float(temperature[0]),
    }


def _segment_sse(
    segment: dict[str, np.ndarray | float],
    dt_s: float,
    inverse_capacity: float,
    cooling_rate: float,
) -> float:
    predicted = euler_thermal_rollout(
        float(segment["temperature0"]),
        segment["heat_W"],
        float(segment["ambient_temperature_K"]),
        dt_s,
        inverse_capacity,
        cooling_rate,
    )
    residual = predicted - np.asarray(segment["temperature_K"], dtype=float)
    return float(residual @ residual)


def _best_thermal_pair(segments, dt_s: float, capacity_axis, cooling_axis):
    best = None
    best_sse = math.inf
    for inverse_capacity in capacity_axis:
        for cooling_rate in cooling_axis:
            sse = 0.0
            for segment in segments:
                sse += _segment_sse(segment, dt_s, float(inverse_capacity), float(cooling_rate))
            if sse < best_sse:
                best_sse = sse
                best = (float(inverse_capacity), float(cooling_rate), sse)
    if best is None:
        raise RuntimeError("thermal search did not produce a candidate")
    return best
