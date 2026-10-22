from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from .config import validate_simulation_config
from .data_io import validate_timeseries
from .schema import SimulationManifest, load_yaml

MODEL_MAP = {
    "SPM": pybamm.lithium_ion.SPM,
    "SPMe": pybamm.lithium_ion.SPMe,
    "DFN": pybamm.lithium_ion.DFN,
}

_TEMPERATURE_VARIABLES = (
    "Volume-averaged cell temperature [K]",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_model(config: dict[str, Any]) -> pybamm.BaseModel:
    model_name = str(config.get("model", "DFN"))
    model_cls = MODEL_MAP[model_name]
    options = dict(config.get("model_options") or {})
    thermal = str(config.get("thermal", "isothermal"))
    options["thermal"] = thermal
    # PyBaMM's default cell geometry is arbitrary. The x-lumped thermal model
    # is formulated for a pouch and rejects that default.
    if thermal == "x-lumped" and "cell geometry" not in options:
        options["cell geometry"] = "pouch"
    return model_cls(options=options)


def _output_time(solution: pybamm.Solution, max_duration_s: float, output_points: int):
    solved = np.asarray(solution.t, dtype=float).squeeze()
    t_stop = float(solved[-1])
    warning = None
    if t_stop < max_duration_s - 1e-6:
        termination = getattr(solution, "termination", "unknown")
        warning = (
            f"solver stopped at {t_stop:.6g} s before max_duration_s={max_duration_s:.6g} s "
            f"({termination})"
        )
    return np.linspace(0.0, t_stop, output_points), warning


def _sample_variable(solution: pybamm.Solution, name: str, t_out: np.ndarray) -> np.ndarray | None:
    try:
        variable = solution[name]
    except KeyError:
        return None
    try:
        values = np.asarray(variable(t_out), dtype=float).squeeze()
    except (TypeError, ValueError, KeyError):
        data = getattr(variable, "data", None)
        if data is None:
            data = getattr(variable, "entries", None)
        if data is None:
            return None
        samples = np.asarray(data, dtype=float).squeeze()
        times = np.asarray(solution.t, dtype=float).squeeze()
        if samples.shape != times.shape:
            return None
        values = np.interp(t_out, times, samples)
    if values.shape != t_out.shape or not np.all(np.isfinite(values)):
        return None
    return values


def _temperature_series(
    solution: pybamm.Solution, t_out: np.ndarray, ambient_temperature_K: float, thermal: str
) -> tuple[np.ndarray, str | None]:
    for name in _TEMPERATURE_VARIABLES:
        values = _sample_variable(solution, name, t_out)
        if values is not None:
            return values, None
    if thermal == "isothermal":
        return np.full(t_out.shape, ambient_temperature_K, dtype=float), (
            "isothermal model did not expose cell temperature; filled with ambient_temperature_K"
        )
    raise RuntimeError(f"thermal model {thermal!r} did not return a finite cell temperature")


def _update_existing(params: pybamm.ParameterValues, updates: dict[str, float]) -> list[str]:
    present = {}
    missing = []
    for key, value in updates.items():
        if key in params:
            present[key] = value
        else:
            missing.append(key)
    if "Current function [A]" not in present:
        raise KeyError("parameter set has no 'Current function [A]'")
    if present:
        params.update(present)
    return [f"parameter not in set, left unchanged: {key}" for key in missing]


def run(config_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    errors = validate_simulation_config(config)
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))

    parameter_set = str(config["parameter_set"])
    params = pybamm.ParameterValues(parameter_set)
    c_rate = float(config.get("c_rate", 1.0))
    ambient_temperature_K = float(config.get("ambient_temperature_K", 298.15))
    initial_soc = float(config.get("initial_soc", 1.0))
    max_duration_s = float(config.get("max_duration_s", 7200))
    output_points = int(config.get("output_points", 500))
    thermal = str(config.get("thermal", "isothermal"))
    try:
        nominal_capacity_Ah = float(params["Nominal cell capacity [A.h]"])
    except KeyError as exc:
        raise KeyError("parameter set has no 'Nominal cell capacity [A.h]'") from exc
    current_A = c_rate * nominal_capacity_Ah
    parameter_warnings = _update_existing(
        params,
        {
            "Current function [A]": current_A,
            "Ambient temperature [K]": ambient_temperature_K,
            "Initial temperature [K]": ambient_temperature_K,
        },
    )

    t_eval = np.linspace(0.0, max_duration_s, output_points)
    model = build_model(config)
    solution = pybamm.Simulation(model, parameter_values=params).solve(
        t_eval=t_eval, initial_soc=initial_soc
    )
    time_s, early_warning = _output_time(solution, max_duration_s, output_points)
    voltage = _sample_variable(solution, "Voltage [V]", time_s)
    if voltage is None:
        raise RuntimeError("solver did not return a finite voltage series")
    temperature, temperature_warning = _temperature_series(
        solution, time_s, ambient_temperature_K, thermal
    )
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "current_A": np.full(time_s.shape, current_A, dtype=float),
            "voltage_V": voltage,
            "temperature_K": temperature,
        }
    )
    data_errors = validate_timeseries(frame)
    if data_errors:
        raise RuntimeError("simulation output failed the data contract: " + "; ".join(data_errors))
    csv_path = out_dir / "timeseries.csv"
    frame.to_csv(csv_path, index=False)

    warnings = [warning for warning in (early_warning, temperature_warning) if warning]
    warnings.extend(parameter_warnings)
    termination = str(getattr(solution, "termination", ""))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = SimulationManifest(
        simulation_id=f"pybamm_{stamp}",
        project_id=str(config.get("project_id") or "battery-ai-comsol-lab"),
        engine="pybamm",
        engine_version=getattr(pybamm, "__version__", None),
        model_id=f"{config['model']}_{thermal}",
        chemistry=str(config["chemistry"]),
        cell_format=str(config.get("cell_format", "generic")),
        parameter_set_id=parameter_set,
        protocol_id=str(config["protocol_id"]),
        commit_sha=_git_sha(),
        config_path=str(config_path),
        status="completed",
        warnings=warnings,
        inputs={
            "c_rate": c_rate,
            "ambient_temperature_K": ambient_temperature_K,
            "initial_soc": initial_soc,
            "nominal_capacity_Ah": nominal_capacity_Ah,
            "current_A": current_A,
            "thermal": thermal,
            "model_options": dict(config.get("model_options") or {}),
            "termination": termination,
            "solved_duration_s": float(time_s[-1]),
        },
        outputs={"timeseries_csv": str(csv_path)},
    )
    manifest_path = out_dir / "manifest.json"
    manifest.write_json(manifest_path)
    return csv_path, manifest_path
