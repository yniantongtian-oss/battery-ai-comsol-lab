from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from .config import validate_simulation_config
from .schema import SimulationManifest, load_yaml

MODEL_MAP = {
    "SPM": pybamm.lithium_ion.SPM,
    "SPMe": pybamm.lithium_ion.SPMe,
    "DFN": pybamm.lithium_ion.DFN,
}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _temperature_entries(solution: pybamm.Solution) -> np.ndarray:
    candidates = (
        "X-averaged cell temperature [K]",
        "Volume-averaged cell temperature [K]",
        "Cell temperature [K]",
    )
    for name in candidates:
        try:
            values = np.asarray(solution[name].entries).squeeze()
            if values.ndim == 1 and values.size == solution.t.size:
                return values
        except KeyError:
            continue
    return np.full(solution.t.shape, np.nan, dtype=float)


def build_model(config: dict[str, Any]) -> pybamm.BaseModel:
    model_name = str(config.get("model", "DFN"))
    model_cls = MODEL_MAP[model_name]
    options = dict(config.get("model_options") or {})
    options["thermal"] = str(config.get("thermal", "isothermal"))
    return model_cls(options=options)


def run(config_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    errors = validate_simulation_config(config)
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))

    model = build_model(config)
    parameter_set = str(config["parameter_set"])
    params = pybamm.ParameterValues(parameter_set)

    c_rate = float(config.get("c_rate", 1.0))
    ambient_temperature_K = float(config.get("ambient_temperature_K", 298.15))
    nominal_capacity_Ah = float(params["Nominal cell capacity [A.h]"])
    current_A = c_rate * nominal_capacity_Ah
    params.update(
        {
            "Current function [A]": current_A,
            "Ambient temperature [K]": ambient_temperature_K,
            "Initial temperature [K]": ambient_temperature_K,
        }
    )

    max_duration_s = float(config.get("max_duration_s", 7200))
    output_points = int(config.get("output_points", 500))
    t_eval = np.linspace(0.0, max_duration_s, output_points)

    simulation = pybamm.Simulation(model, parameter_values=params)
    solution = simulation.solve(t_eval=t_eval, initial_soc=float(config.get("initial_soc", 1.0)))

    time_s = np.asarray(solution.t).squeeze()
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "current_A": np.full(time_s.shape, current_A, dtype=float),
            "voltage_V": np.asarray(solution["Voltage [V]"].entries).squeeze(),
            "temperature_K": _temperature_entries(solution),
        }
    )
    csv_path = out_dir / "timeseries.csv"
    frame.to_csv(csv_path, index=False)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = SimulationManifest(
        simulation_id=f"pybamm_{stamp}",
        engine="pybamm",
        engine_version=getattr(pybamm, "__version__", None),
        model_id=f"{config['model']}_{config.get('thermal', 'isothermal')}",
        chemistry=str(config["chemistry"]),
        cell_format=str(config.get("cell_format", "generic")),
        parameter_set_id=parameter_set,
        protocol_id=str(config["protocol_id"]),
        commit_sha=_git_sha(),
        config_path=str(config_path),
        status="completed",
        warnings=[str(getattr(solution, "termination", ""))],
        inputs={
            "c_rate": c_rate,
            "ambient_temperature_K": ambient_temperature_K,
            "initial_soc": float(config.get("initial_soc", 1.0)),
            "nominal_capacity_Ah": nominal_capacity_Ah,
            "current_A": current_A,
        },
        outputs={"timeseries_csv": str(csv_path)},
    )
    manifest_path = out_dir / "manifest.json"
    manifest.write_json(manifest_path)
    return csv_path, manifest_path
