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

    time = frame["time_s"].to_numpy(dtype=float)
    if np.any(~np.isfinite(time)):
        errors.append("time_s contains non-finite values")
    if np.any(np.diff(time) < 0):
        errors.append("time_s must be monotonically non-decreasing")

    voltage = frame["voltage_V"].to_numpy(dtype=float)
    if np.any(~np.isfinite(voltage)):
        errors.append("voltage_V contains non-finite values")
    if np.nanmin(voltage) < -1 or np.nanmax(voltage) > 10:
        errors.append("voltage_V contains values outside broad Li-ion sanity bounds")

    temperature = frame["temperature_K"].to_numpy(dtype=float)
    finite_temperature = temperature[np.isfinite(temperature)]
    if finite_temperature.size and (
        np.nanmin(finite_temperature) < 150 or np.nanmax(finite_temperature) > 500
    ):
        errors.append("temperature_K contains values outside supported sanity bounds")
    return errors


def read_timeseries(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))
