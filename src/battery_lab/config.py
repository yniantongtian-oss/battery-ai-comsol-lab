from __future__ import annotations

from pathlib import Path
from typing import Any

from .numbers import finite_float, strict_int
from .schema import load_yaml

VALID_MODELS = {"SPM", "SPMe", "DFN"}
VALID_THERMAL = {"isothermal", "lumped", "x-lumped", "x-full"}
CONFIG_KINDS = {"simulation", "sweep", "pack", "target-cell"}


def detect_config_kind(config: dict[str, Any]) -> str:
    """Infer the repository configuration type from its top-level keys."""
    api_version = config.get("api_version")
    if isinstance(api_version, str) and api_version.startswith("battery-lab/spec/"):
        return "spec"
    if "target_id" in config or ("evidence" in config and "reference_simulation" in config):
        return "target-cell"
    if "base_config" in config and ("c_rates" in config or "ambient_temperatures_K" in config):
        return "sweep"
    if "cell_config" in config and ("series_cells" in config or "parallel_cells" in config):
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

    _check_range(
        config,
        "c_rate",
        1.0,
        errors,
        lambda value: value > 0,
        "c_rate must be a finite number > 0",
    )
    _check_range(
        config,
        "ambient_temperature_K",
        298.15,
        errors,
        lambda value: 150 <= value <= 500,
        "ambient_temperature_K must be a finite number between 150 and 500",
    )
    _check_range(
        config,
        "initial_soc",
        1.0,
        errors,
        lambda value: 0 <= value <= 1,
        "initial_soc must be a finite number between 0 and 1",
    )
    _check_range(
        config,
        "max_duration_s",
        7200,
        errors,
        lambda value: value > 0,
        "max_duration_s must be a finite number > 0",
    )
    if "output_points" not in config:
        output_points = 500
    else:
        try:
            output_points = strict_int(config["output_points"])
        except ValueError:
            errors.append("output_points must be an integer >= 2")
            output_points = None
    if output_points is not None and output_points < 2:
        errors.append("output_points must be an integer >= 2")

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
                number = finite_float(value)
            except ValueError:
                errors.append(f"{key} contains a non-finite or non-numeric value: {value!r}")
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
            value = strict_int(config.get(key, 0))
            if value <= 0:
                errors.append(f"{key} must be a positive integer")
        except ValueError:
            errors.append(f"{key} must be a positive integer")

    for key in ("initial_soc_mean", "initial_soc_std"):
        if key not in config:
            continue
        try:
            value = finite_float(config[key])
            if key == "initial_soc_mean" and not 0 <= value <= 1:
                errors.append("initial_soc_mean must be a finite number between 0 and 1")
            if key == "initial_soc_std" and value < 0:
                errors.append("initial_soc_std must be a finite number >= 0")
        except ValueError:
            errors.append(f"{key} must be a finite number")
    return errors


def validate_config(config: dict[str, Any], config_path: str | Path | None = None) -> list[str]:
    kind = detect_config_kind(config)
    if kind == "spec":
        from .spec import validate_spec

        return validate_spec(config, config_path)
    if kind == "target-cell":
        from .target_cell import validate_target_cell_spec

        root = Path(config_path).resolve().parent.parent if config_path is not None else None
        return validate_target_cell_spec(config, root=root)
    if kind == "sweep":
        return validate_sweep_config(config, config_path)
    if kind == "pack":
        return validate_pack_config(config, config_path)
    return validate_simulation_config(config)


def validate_config_file(path: str | Path) -> list[str]:
    path = Path(path)
    return validate_config(load_yaml(path), path)


def _check_range(config, key, default, errors, predicate, message) -> None:
    raw = config.get(key, default)
    try:
        value = finite_float(raw)
    except ValueError:
        errors.append(message)
        return
    if not predicate(value):
        errors.append(message)
