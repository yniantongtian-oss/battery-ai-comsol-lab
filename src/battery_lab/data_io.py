from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_TIMESERIES_COLUMNS = ("time_s", "current_A", "voltage_V", "temperature_K")


def validate_timeseries(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [column for column in REQUIRED_TIMESERIES_COLUMNS if column not in frame.columns]
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
        return errors

    if frame.empty:
        errors.append("dataset is empty")
        return errors

    values: dict[str, np.ndarray] = {}
    for column in REQUIRED_TIMESERIES_COLUMNS:
        try:
            array = frame[column].to_numpy(dtype=float)
        except (ValueError, TypeError):
            errors.append(f"{column} contains non-numeric values")
            continue
        values[column] = array
        if np.any(~np.isfinite(array)):
            errors.append(f"{column} contains non-finite values")

    time = values.get("time_s", np.array([]))
    if np.any(np.diff(time) < 0):
        errors.append("time_s must be monotonically non-decreasing")

    voltage = values.get("voltage_V", np.array([]))
    finite_voltage = voltage[np.isfinite(voltage)]
    if finite_voltage.size and (np.min(finite_voltage) < -1 or np.max(finite_voltage) > 10):
        errors.append("voltage_V contains values outside broad Li-ion sanity bounds")

    temperature = values.get("temperature_K", np.array([]))
    finite_temperature = temperature[np.isfinite(temperature)]
    if finite_temperature.size and (
        np.nanmin(finite_temperature) < 150 or np.nanmax(finite_temperature) > 500
    ):
        errors.append("temperature_K contains values outside supported sanity bounds")
    return errors


def read_timeseries(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))
