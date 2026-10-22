"""YAML specs for agent-driven simulation and surrogate benchmarks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import validate_simulation_config
from .numbers import finite_float, strict_int
from .schema import load_yaml

SPEC_API_VERSION = "battery-lab/spec/v1"
SIMULATION_KINDS = {"simulation", "surrogate-benchmark"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_SIMULATION_KEYS = {
    "api_version",
    "kind",
    "name",
    "output_dir",
    "base_config",
    "simulation",
    "overrides",
    "notes",
}
_BENCHMARK_KEYS = {"api_version", "kind", "name", "output_dir", "seed", "benchmark", "notes"}
_OVERRIDE_KEYS = {
    "project_id",
    "chemistry",
    "cell_format",
    "engine",
    "model",
    "thermal",
    "parameter_set",
    "protocol_id",
    "c_rate",
    "ambient_temperature_K",
    "initial_soc",
    "max_duration_s",
    "output_points",
    "model_options",
    "notes",
}
_BENCHMARK_FIELDS = {
    "sample_dt_s",
    "soc_start",
    "soc_stop",
    "voltage_noise_std_V",
    "temperature_noise_std_K",
    "ridge_alpha",
    "tau_min_s",
    "tau_max_s",
    "tau_count",
    "train_protocols",
    "test_protocols",
    "thresholds",
}
_THRESHOLD_FIELDS = {
    "physics_test_voltage_rmse_V",
    "thermal_rollout_rmse_K",
    "minimum_rmse_ratio_black_box_over_physics",
}


def load_spec(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def load_spec_text(text: str) -> dict[str, Any]:
    if len(text) > 200_000:
        raise ValueError("spec text exceeds 200000 characters")
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("spec text must be a YAML mapping")
    return data


def validate_spec(spec: dict[str, Any], spec_path: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if spec.get("api_version") != SPEC_API_VERSION:
        errors.append(f"api_version must be {SPEC_API_VERSION}")
    kind = spec.get("kind")
    if kind not in SIMULATION_KINDS:
        errors.append("kind must be simulation or surrogate-benchmark")
    name = spec.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        errors.append("name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,80}")
    _check_relative_dir(spec.get("output_dir"), errors)
    allowed = _SIMULATION_KEYS if kind == "simulation" else _BENCHMARK_KEYS
    if kind in SIMULATION_KINDS:
        unknown = sorted(set(spec) - allowed)
        if unknown:
            errors.append("unknown spec fields: " + ", ".join(unknown))
    if kind == "simulation":
        errors.extend(_validate_simulation_spec(spec, spec_path))
    elif kind == "surrogate-benchmark":
        errors.extend(_validate_benchmark_spec(spec))
    return errors


def validate_spec_file(path: str | Path) -> list[str]:
    path = Path(path)
    return validate_spec(load_spec(path), path)


def resolve_simulation_config(
    spec: dict[str, Any], spec_path: str | Path | None = None
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = _validate_simulation_spec(spec, spec_path)
    if errors:
        return None, errors
    if spec.get("base_config"):
        resolved = _resolve_existing(spec_path, str(spec["base_config"]))
        config = load_yaml(resolved) if resolved is not None else {}
    else:
        config = dict(spec["simulation"])
    config.update(spec.get("overrides") or {})
    return config, []


def run_spec(
    spec: dict[str, Any],
    *,
    spec_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    working = dict(spec)
    if output_dir is not None:
        working["output_dir"] = str(output_dir)
    if output_dir is None:
        errors = validate_spec(working, spec_path)
    else:
        errors = _errors_with_output_override(working, spec_path)
    kind = str(working.get("kind") or "unknown")
    name = str(working.get("name") or "")
    if errors:
        return _envelope(ok=False, kind=kind, name=name, errors=errors, dry_run=dry_run)
    if kind == "surrogate-benchmark":
        return _run_benchmark_spec(working, dry_run=dry_run)
    return _run_simulation_spec(working, spec_path=spec_path, dry_run=dry_run)


def run_spec_file(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    spec_path = Path(path)
    try:
        spec = load_spec(spec_path)
    except (OSError, ValueError) as exc:
        return _envelope(
            ok=False,
            kind="unknown",
            name=spec_path.stem,
            errors=[str(exc)],
            dry_run=dry_run,
        )
    result = run_spec(spec, spec_path=spec_path, output_dir=output_dir, dry_run=dry_run)
    if result["ok"] and not dry_run:
        destination = Path(result["artifacts"]["output_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / "result.json"
        result["artifacts"]["result_json"] = str(result_path)
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def _errors_with_output_override(spec: dict[str, Any], spec_path: str | Path | None) -> list[str]:
    """Validate a spec whose output directory was replaced by the caller."""
    original = spec.get("output_dir")
    probe = dict(spec)
    probe["output_dir"] = "runs/agent-output"
    errors = [
        error for error in validate_spec(probe, spec_path) if not error.startswith("output_dir")
    ]
    if not isinstance(original, str) or not original.strip() or "\x00" in original:
        errors.append("output_dir must be a non-empty path")
    return errors


def _run_simulation_spec(
    spec: dict[str, Any], *, spec_path: str | Path | None, dry_run: bool
) -> dict[str, Any]:
    config, errors = resolve_simulation_config(spec, spec_path)
    if errors or config is None:
        return _envelope(
            ok=False,
            kind="simulation",
            name=str(spec["name"]),
            errors=errors,
            dry_run=dry_run,
        )
    output = Path(str(spec["output_dir"]))
    if dry_run:
        return _envelope(
            ok=True,
            kind="simulation",
            name=str(spec["name"]),
            errors=[],
            dry_run=True,
            resolved_config=config,
            artifacts={"output_dir": str(output)},
        )
    try:
        import importlib.util

        if importlib.util.find_spec("pybamm") is None:
            return _envelope(
                ok=False,
                kind="simulation",
                name=str(spec["name"]),
                errors=["optional runtime dependency 'pybamm' is not installed"],
                dry_run=False,
                resolved_config=config,
                extra={"install": 'python -m pip install -e ".[pybamm]"'},
            )
        from .pybamm_runner import run

        config_path = output / "config.yaml"
        output.mkdir(parents=True, exist_ok=True)
        import yaml

        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        csv_path, manifest_path = run(config_path, output)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        return _envelope(
            ok=False,
            kind="simulation",
            name=str(spec["name"]),
            errors=[str(exc)],
            dry_run=False,
            resolved_config=config,
        )
    import pandas as pd

    frame = pd.read_csv(csv_path)
    return _envelope(
        ok=True,
        kind="simulation",
        name=str(spec["name"]),
        errors=[],
        dry_run=False,
        resolved_config=config,
        artifacts={
            "output_dir": str(output),
            "timeseries_csv": str(csv_path),
            "manifest_json": str(manifest_path),
            "config_yaml": str(config_path),
        },
        metrics={
            "samples": int(len(frame)),
            "duration_s": float(frame["time_s"].iloc[-1]),
            "end_voltage_V": float(frame["voltage_V"].iloc[-1]),
            "min_voltage_V": float(frame["voltage_V"].min()),
            "max_temperature_K": float(frame["temperature_K"].max()),
        },
    )


def _run_benchmark_spec(spec: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    output = Path(str(spec["output_dir"]))
    if dry_run:
        return _envelope(
            ok=True,
            kind="surrogate-benchmark",
            name=str(spec["name"]),
            errors=[],
            dry_run=True,
            resolved_config=spec["benchmark"],
            artifacts={"output_dir": str(output)},
        )
    from .benchmark import run_benchmark

    try:
        report = run_benchmark(spec, output)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        return _envelope(
            ok=False,
            kind="surrogate-benchmark",
            name=str(spec["name"]),
            errors=[str(exc)],
            dry_run=False,
        )
    physics = report["models"]["physics_informed"]["voltage_V"]["test"]["rmse"]
    black = report["models"]["black_box_ridge"]["voltage_V"]["test"]["rmse"]
    thermal = report["models"]["physics_informed"]["temperature_K"]["test"]["rmse"]
    return _envelope(
        ok=bool(report["passed"]),
        kind="surrogate-benchmark",
        name=str(spec["name"]),
        errors=list(report["failures"]),
        dry_run=False,
        artifacts={
            "output_dir": str(output),
            "benchmark_report_json": str(output / "benchmark_report.json"),
            "train_csv": str(output / "train.csv"),
            "test_csv": str(output / "test.csv"),
        },
        metrics={
            "passed": bool(report["passed"]),
            "physics_test_voltage_rmse_V": physics,
            "black_box_test_voltage_rmse_V": black,
            "thermal_rollout_rmse_K": thermal,
            "test_voltage_rmse_ratio_black_box_over_physics": report["comparison"][
                "test_voltage_rmse_ratio_black_box_over_physics"
            ],
            "train_sha256": report["dataset"]["train_sha256"],
            "test_sha256": report["dataset"]["test_sha256"],
            "metrics_sha256": report["metrics_sha256"],
        },
    )


def _validate_simulation_spec(spec: dict[str, Any], spec_path: str | Path | None) -> list[str]:
    errors: list[str] = []
    has_base = bool(spec.get("base_config"))
    has_inline = "simulation" in spec
    if has_base == has_inline:
        errors.append("provide exactly one of base_config or simulation")
        return errors
    overrides = spec.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        errors.append("overrides must be a mapping")
        return errors
    unknown = sorted(set(overrides) - _OVERRIDE_KEYS)
    if unknown:
        errors.append("unknown override fields: " + ", ".join(unknown))
    model_options = overrides.get("model_options")
    if model_options is not None and not _string_keyed_scalars(model_options):
        errors.append("overrides.model_options must map strings to numbers, strings, or booleans")
    if has_base:
        resolved = _resolve_existing(spec_path, str(spec["base_config"]))
        if resolved is None:
            errors.append(f"base_config does not exist: {spec['base_config']}")
            return errors
        if unknown:
            return errors
        config = load_yaml(resolved)
        config.update(overrides)
        errors.extend(validate_simulation_config(config))
        return errors
    simulation = spec.get("simulation")
    if not isinstance(simulation, dict):
        errors.append("simulation must be a mapping")
        return errors
    if unknown:
        return errors
    merged = dict(simulation)
    merged.update(overrides)
    errors.extend(validate_simulation_config(merged))
    return errors


def _validate_benchmark_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        seed = strict_int(spec.get("seed"))
        if not 0 <= seed <= 2**32 - 1:
            errors.append("seed must be an integer between 0 and 2**32 - 1")
    except (TypeError, ValueError):
        errors.append("seed must be an integer between 0 and 2**32 - 1")
    benchmark = spec.get("benchmark")
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be a mapping")
        return errors
    unknown = sorted(set(benchmark) - _BENCHMARK_FIELDS)
    if unknown:
        errors.append("unknown benchmark fields: " + ", ".join(unknown))
    missing = sorted(_BENCHMARK_FIELDS - set(benchmark))
    if missing:
        errors.append("missing benchmark fields: " + ", ".join(missing))
        return errors
    _check_positive(benchmark, "sample_dt_s", errors)
    _check_positive(benchmark, "tau_min_s", errors)
    _check_positive(benchmark, "tau_max_s", errors)
    try:
        if finite_float(benchmark["tau_max_s"]) <= finite_float(benchmark["tau_min_s"]):
            errors.append("tau_max_s must be greater than tau_min_s")
    except ValueError:
        pass
    try:
        count = strict_int(benchmark["tau_count"])
        if count < 2:
            errors.append("tau_count must be an integer >= 2")
    except (TypeError, ValueError):
        errors.append("tau_count must be an integer >= 2")
    try:
        soc_start = finite_float(benchmark["soc_start"])
        soc_stop = finite_float(benchmark["soc_stop"])
        if not 0.0 < soc_stop < soc_start <= 1.0:
            errors.append("require 0 < soc_stop < soc_start <= 1")
    except ValueError:
        errors.append("soc_start and soc_stop must be finite numbers")
    for key in ("voltage_noise_std_V", "temperature_noise_std_K", "ridge_alpha"):
        try:
            if finite_float(benchmark[key]) < 0:
                errors.append(f"{key} must be >= 0")
        except ValueError:
            errors.append(f"{key} must be a finite number")
    train = _validate_protocols(benchmark.get("train_protocols"), "train_protocols", errors)
    test = _validate_protocols(benchmark.get("test_protocols"), "test_protocols", errors)
    overlap = sorted(set(train) & set(test))
    if overlap:
        errors.append("test protocol repeats a training protocol: " + ", ".join(overlap))
    try:
        dt_s = finite_float(benchmark["sample_dt_s"])
        soc_start = finite_float(benchmark["soc_start"])
        soc_stop = finite_float(benchmark["soc_stop"])
    except ValueError:
        dt_s = None
    if dt_s is not None and dt_s > 0:
        for label, protocols in (
            ("train_protocols", benchmark["train_protocols"]),
            ("test_protocols", benchmark["test_protocols"]),
        ):
            if not isinstance(protocols, list):
                continue
            for protocol in protocols:
                if not isinstance(protocol, dict) or "c_rate" not in protocol:
                    continue
                try:
                    c_rate = finite_float(protocol["c_rate"])
                except ValueError:
                    continue
                if c_rate <= 0:
                    continue
                samples = int((soc_start - soc_stop) / c_rate * 3600.0 / dt_s)
                if samples < 5:
                    errors.append(f"{label} protocol {protocol!r} has fewer than 5 samples")
    thresholds = benchmark.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("thresholds must be a mapping")
    else:
        missing_thresholds = sorted(_THRESHOLD_FIELDS - set(thresholds))
        if missing_thresholds:
            errors.append("missing thresholds: " + ", ".join(missing_thresholds))
        for key in _THRESHOLD_FIELDS:
            if key not in thresholds:
                continue
            try:
                if finite_float(thresholds[key]) <= 0:
                    errors.append(f"thresholds.{key} must be > 0")
            except ValueError:
                errors.append(f"thresholds.{key} must be a finite number")
    return errors


def _validate_protocols(value: Any, label: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a non-empty list")
        return []
    keys: list[str] = []
    for protocol in value:
        if not isinstance(protocol, dict):
            errors.append(f"{label} entries must be mappings")
            continue
        unknown = sorted(set(protocol) - {"c_rate", "ambient_temperature_K"})
        if unknown:
            errors.append(f"{label} has unknown fields: " + ", ".join(unknown))
        try:
            c_rate = finite_float(protocol.get("c_rate"))
            if c_rate <= 0:
                errors.append(f"{label} c_rate must be > 0")
        except (TypeError, ValueError):
            errors.append(f"{label} c_rate must be a finite number")
            c_rate = None
        try:
            ambient = finite_float(protocol.get("ambient_temperature_K"))
            if not 150.0 <= ambient <= 500.0:
                errors.append(f"{label} ambient_temperature_K must be between 150 and 500 K")
        except (TypeError, ValueError):
            errors.append(f"{label} ambient_temperature_K must be a finite number")
            ambient = None
        if c_rate is not None and ambient is not None:
            key = f"C{c_rate:.3f}_T{ambient:.2f}"
            if key in keys:
                errors.append(f"{label} contains duplicate protocol {key}")
            keys.append(key)
    return keys


def _check_positive(mapping: dict[str, Any], key: str, errors: list[str]) -> None:
    try:
        if finite_float(mapping[key]) <= 0:
            errors.append(f"{key} must be > 0")
    except (TypeError, ValueError):
        errors.append(f"{key} must be a finite number")


def _check_relative_dir(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append("output_dir must be a non-empty relative path")
        return
    if "\x00" in value or Path(value).is_absolute():
        errors.append("output_dir must be a non-empty relative path")


def _resolve_existing(spec_path: str | Path | None, raw: str) -> Path | None:
    """Find a spec path from the spec file, its repository root, or the working directory."""
    candidate = Path(raw)
    options: list[Path] = []
    if candidate.is_absolute():
        options.append(candidate)
    elif spec_path is not None:
        spec_dir = Path(spec_path).resolve().parent
        options.append(spec_dir / candidate)
        options.append(spec_dir.parent / candidate)
        options.append(Path.cwd() / candidate)
    else:
        options.append(Path.cwd() / candidate)
    for option in options:
        if option.is_file():
            return option
    return None


def _string_keyed_scalars(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) and _scalar(item) for key, item in value.items())


def _scalar(value: Any) -> bool:
    if isinstance(value, bool | int | float | str):
        return True
    return False


def _envelope(
    *,
    ok: bool,
    kind: str,
    name: str,
    errors: list[str],
    dry_run: bool,
    artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
    resolved_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "kind": kind,
        "name": name,
        "dry_run": dry_run,
        "errors": errors,
        "artifacts": artifacts or {},
        "metrics": metrics or {},
        "resolved_config": resolved_config,
    }
    if extra:
        payload.update(extra)
    return payload
