from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import load_yaml

_REQUIRED_EVIDENCE = (
    "datasheet",
    "capacity_reference",
    "ocv_soc",
    "hppc",
    "thermal",
)
_ALLOWED_STATUS = {"required", "available", "validated"}


def validate_target_cell_spec(
    spec: dict[str, Any], *, root: str | Path | None = None
) -> list[str]:
    """Validate a Phase-B target-cell definition without inventing missing evidence."""
    errors: list[str] = []
    for key in (
        "target_id",
        "cell_revision",
        "chemistry",
        "form_factor",
        "parameter_set",
        "reference_simulation",
    ):
        if not spec.get(key):
            errors.append(f"missing required field: {key}")

    base = Path(root) if root is not None else None
    reference = spec.get("reference_simulation")
    if reference and base is not None:
        ref_path = Path(str(reference))
        if not ref_path.is_absolute():
            ref_path = base / ref_path
        if not ref_path.exists():
            errors.append(f"reference_simulation does not exist: {reference}")

    evidence = spec.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be a mapping")
        evidence = {}

    for name in _REQUIRED_EVIDENCE:
        item = evidence.get(name)
        if not isinstance(item, dict):
            errors.append(f"missing evidence entry: {name}")
            continue
        status = item.get("status")
        if status not in _ALLOWED_STATUS:
            errors.append(
                f"evidence.{name}.status must be one of {sorted(_ALLOWED_STATUS)}"
            )
        if status in {"available", "validated"}:
            path = item.get("path")
            digest = item.get("sha256")
            if not path:
                errors.append(f"evidence.{name}.path required when status={status}")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(
                    f"evidence.{name}.sha256 must be a 64-character digest when status={status}"
                )
            if path and base is not None:
                data_path = Path(str(path))
                if not data_path.is_absolute():
                    data_path = base / data_path
                if not data_path.exists():
                    errors.append(f"evidence.{name}.path does not exist: {path}")

    protocols = spec.get("protocols")
    if not isinstance(protocols, dict):
        errors.append("protocols must be a mapping")
    else:
        for name in _REQUIRED_EVIDENCE[1:]:
            if not protocols.get(name):
                errors.append(f"missing protocol id: {name}")

    replicates = spec.get("replicates", {})
    try:
        minimum = int(replicates.get("minimum_per_condition", 0))
        if minimum < 1:
            errors.append("replicates.minimum_per_condition must be >= 1")
    except (AttributeError, TypeError, ValueError):
        errors.append("replicates.minimum_per_condition must be an integer >= 1")

    thresholds = spec.get("acceptance_thresholds", {})
    if not isinstance(thresholds, dict):
        errors.append("acceptance_thresholds must be a mapping")
    else:
        for name, value in thresholds.items():
            if value is None:
                continue
            try:
                if float(value) <= 0:
                    errors.append(f"acceptance_thresholds.{name} must be > 0 or null")
            except (TypeError, ValueError):
                errors.append(f"acceptance_thresholds.{name} must be numeric or null")

    return errors


def load_target_cell_spec(path: str | Path) -> dict[str, Any]:
    return load_yaml(Path(path))


def evidence_readiness(spec: dict[str, Any]) -> dict[str, str]:
    evidence = spec.get("evidence", {})
    return {
        name: str(evidence.get(name, {}).get("status", "missing"))
        for name in _REQUIRED_EVIDENCE
    }
