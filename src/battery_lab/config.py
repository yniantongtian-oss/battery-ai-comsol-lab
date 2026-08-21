from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import load_yaml

VALID_MODELS = {"SPM", "SPMe", "DFN"}
VALID_THERMAL = {"isothermal", "lumped", "x-lumped", "x-full"}
CONFIG_KINDS = {"simulation", "sweep", "pack"}


def detect_config_kind(config: dict[str, Any]) -> str:
    """Infer the repository configuration type from its top-level keys."""
    if "base_config" in config and (
        "c_rates" in config or "ambient_temperatures_K" in config
    ):
        return "sweep"
    if "cell_config" in config and (
        "series_cells" in config or "parallel_cells" in config
    ):
        return "pack"
    return "simulation"


def validate_simulation_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["chemistry", "engine", "model", "parameter_set", "protocol_id"]
    for key in required:
        if not config.get(key):
            errors.append(f"missing required field: {key}")

    engine = str(config.get("engine", ""))
    if engine and engine != "pybamm":
        errors.append(f"unsupported simulation engine: {engine}")

    model = str(config.get("model", ""))
    if model and model not in VALID_MODELS:
        errors.append(f"unsupported model: {model}")

    thermal = str(config.get("thermal", "isothermal"))
    if thermal not in VALID_THERMAL:
        errors.append(f"unsupported thermal option: {thermal}")

    try:
        c_rate = float(config.get("c_rate", 1.0))
        if c_rate <= 0:
            errors.append("c_rate must be > 0")
    except (TypeError, ValueError):
        errors.append("c_rate must be numeric")

    try:
        temperature = float(config.get("ambient_temperature_K", 298.15))
        if not 150 <= temperature <= 500:
            errors.append("ambient_temperature_K outside supported range")
    except (TypeError, ValueError):
        errors.append("ambient_temperature_K must be numeric")

    try:
        initial_soc = float(config.get("initial_soc", 1.0))
        if not 0 <= initial_soc <= 1:
            errors.append("initial_soc must be between 0 and 1")
    except (TypeError, ValueError):
        errors.append("initial_soc must be numeric")

    try:
        duration = float(config.get("max_duration_s", 7200))
        if duration <= 0:
            errors.append("max_duration_s must be > 0")
    except (TypeError, ValueError):
        errors.append("max_duration_s must be numeric")

    try:
        output_points = int(config.get("output_points", 500))
        if output_points < 2:
            errors.append("output_points must be >= 2")
    except (TypeError, ValueError):
        errors.append("output_points must be an integer")

    return errors


def validate_sweep_config(
    config: dict[str, Any], config_path: str | Path | None = None
) -> list[str]:
    errors: list[str] = []
    base_config = config.get("base_config")
    if not base_config:
        errors.append("missing required field: base_config")
    elif config_path is not None:
        path = Path(config_path)
        resolved = Path(str(base_config))
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        if not resolved.exists():
            errors.append(f"base_config does not exist: {base_config}")

    for key in ("c_rates", "ambient_temperatures_K"):
        values = config.get(key)
        if not isinstance(values, list) or not values:
            errors.append(f"{key} must be a non-empty list")
            continue
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{key} contains a non-numeric value: {value!r}")
                continue
            if key == "c_rates" and number <= 0:
                errors.append("c_rates values must be > 0")
            if key == "ambient_temperatures_K" and not 150 <= number <= 500:
                errors.append("ambient_temperatures_K values must be between 150 and 500 K")
    return errors


def validate_pack_config(
    config: dict[str, Any], config_path: str | Path | None = None
) -> list[str]:
    errors: list[str] = []
    cell_config = config.get("cell_config")
    if not cell_config:
        errors.append("missing required field: cell_config")
    elif config_path is not None:
        path = Path(config_path)
        resolved = Path(str(cell_config))
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        if not resolved.exists():
            errors.append(f"cell_config does not exist: {cell_config}")

    for key in ("series_cells", "parallel_cells"):
        try:
            value = int(config.get(key, 0))
            if value <= 0:
                errors.append(f"{key} must be a positive integer")
        except (TypeError, ValueError):
            errors.append(f"{key} must be a positive integer")

    for key in ("initial_soc_mean", "initial_soc_std"):
        if key not in config:
            continue
        try:
            value = float(config[key])
            if key == "initial_soc_mean" and not 0 <= value <= 1:
                errors.append("initial_soc_mean must be between 0 and 1")
            if key == "initial_soc_std" and value < 0:
                errors.append("initial_soc_std must be >= 0")
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric")
    return errors


def validate_config(config: dict[str, Any], config_path: str | Path | None = None) -> list[str]:
    kind = detect_config_kind(config)
    if kind == "sweep":
        return validate_sweep_config(config, config_path)
    if kind == "pack":
        return validate_pack_config(config, config_path)
    return validate_simulation_config(config)


def validate_config_file(path: str | Path) -> list[str]:
    path = Path(path)
    return validate_config(load_yaml(path), path)
