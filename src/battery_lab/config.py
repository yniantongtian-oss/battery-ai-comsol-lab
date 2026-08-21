from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import load_yaml


VALID_MODELS = {"SPM", "SPMe", "DFN"}
VALID_THERMAL = {"isothermal", "lumped", "x-lumped", "x-full"}


def validate_simulation_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["chemistry", "engine", "model", "parameter_set", "protocol_id"]
    for key in required:
        if not config.get(key):
            errors.append(f"missing required field: {key}")

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

    return errors


def validate_config_file(path: str | Path) -> list[str]:
    return validate_simulation_config(load_yaml(path))
