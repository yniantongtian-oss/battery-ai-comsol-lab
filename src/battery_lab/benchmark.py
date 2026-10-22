"""Reproducible physics-informed surrogate benchmark.

The hidden plant is a rested constant-current Thevenin model with Arrhenius
ohmic and RC resistances, a cubic open-circuit voltage, and a lumped thermal
energy balance. Nominal capacity and activation energy are shared literature
priors. Resistances, the open-circuit-voltage coefficients, the RC time
constant, and the thermal parameters are not shown to the fitting code.

Training and test rows are split by protocol. The unstructured baseline is a
standardized ridge on time, current, and ambient temperature. Metrics are a
pure function of the spec seed and are written without timestamps or absolute
paths so two runs produce the same report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .metrics import mae, max_abs_error, rmse
from .surrogate import (
    NOMINAL_CAPACITY_AH,
    GreyBoxTheveninSurrogate,
    LumpedThermalSurrogate,
    ProtocolRidgeSurrogate,
    electrical_heat_W,
    euler_thermal_rollout,
    grey_box_voltage,
    ocv_derivative_min,
)

_TRUTH = {
    "r0_ohm": 0.02,
    "r1_ohm": 0.025,
    "tau_s": 47.2,
    "ocv_coefficients": (2.9, 0.7, 0.1, 0.3),
    "thermal_capacity_J_per_K": 75.0,
    "heat_transfer_W_per_K": 0.55,
}

DATASET_COLUMNS = (
    "time_s",
    "current_A",
    "ambient_temperature_K",
    "voltage_V",
    "temperature_K",
    "protocol_id",
)


def canonical_csv(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, list(DATASET_COLUMNS)]
    return ordered.to_csv(index=False, float_format="%.10e", lineterminator="\n")


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_benchmark(spec: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    benchmark = spec["benchmark"]
    train, test = _generate_datasets(spec)
    train_csv = canonical_csv(train)
    test_csv = canonical_csv(test)
    (output / "train.csv").write_text(train_csv, encoding="utf-8")
    (output / "test.csv").write_text(test_csv, encoding="utf-8")

    report = _score(train, test, benchmark)
    report["benchmark_id"] = str(spec["name"])
    report["library_version"] = __version__
    report["seed"] = int(spec["seed"])
    report["dataset"] = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sha256": sha256_text(train_csv),
        "test_sha256": sha256_text(test_csv),
        "train_protocols": _protocol_table(benchmark["train_protocols"]),
        "test_protocols": _protocol_table(benchmark["test_protocols"]),
        "split": "protocol",
    }
    report["assumptions"] = {
        "nominal_capacity_Ah": NOMINAL_CAPACITY_AH,
        "activation_energy_J_per_mol": 25_000.0,
        "reference_temperature_K": 298.15,
        "electrical_model": "cubic OCV, Arrhenius R0 and R1, one RC pair from rest",
        "thermal_model": "lumped energy balance with Euler rollout",
        "black_box": "standardized ridge on time_s, current_A, ambient_temperature_K",
        "comsol_used": False,
    }
    failures = _threshold_failures(report, benchmark["thresholds"])
    report["thresholds"] = _plain_thresholds(benchmark["thresholds"])
    report["failures"] = failures
    report["passed"] = not failures
    metrics_payload = {
        "models": report["models"],
        "comparison": report["comparison"],
        "dataset": {
            "train_sha256": report["dataset"]["train_sha256"],
            "test_sha256": report["dataset"]["test_sha256"],
        },
        "passed": report["passed"],
    }
    report["metrics_sha256"] = sha256_text(
        json.dumps(metrics_payload, sort_keys=True, separators=(",", ":"))
    )
    report_path = output / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _generate_datasets(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = spec["benchmark"]
    rng = np.random.default_rng(int(spec["seed"]))
    train = _protocols_frame(benchmark["train_protocols"], benchmark, rng)
    test = _protocols_frame(benchmark["test_protocols"], benchmark, rng)
    return train, test


def _protocols_frame(
    protocols: list[dict[str, Any]], benchmark: dict[str, Any], rng
) -> pd.DataFrame:
    frames = [_one_protocol(protocol, benchmark, rng) for protocol in protocols]
    return pd.concat(frames, ignore_index=True)


def _one_protocol(protocol: dict[str, Any], benchmark: dict[str, Any], rng) -> pd.DataFrame:
    c_rate = float(protocol["c_rate"])
    ambient = float(protocol["ambient_temperature_K"])
    dt_s = float(benchmark["sample_dt_s"])
    soc_start = float(benchmark["soc_start"])
    soc_stop = float(benchmark["soc_stop"])
    duration_s = (soc_start - soc_stop) / c_rate * 3600.0
    sample_count = int(np.floor(duration_s / dt_s)) + 1
    time_s = np.arange(sample_count, dtype=float) * dt_s
    current_A = np.full(sample_count, c_rate * NOMINAL_CAPACITY_AH)
    soc = soc_start - current_A * time_s / (NOMINAL_CAPACITY_AH * 3600.0)
    voltage = grey_box_voltage(
        time_s,
        current_A,
        ambient,
        soc,
        _TRUTH["r0_ohm"],
        _TRUTH["r1_ohm"],
        _TRUTH["tau_s"],
        _TRUTH["ocv_coefficients"],
    )
    heat = electrical_heat_W(
        time_s,
        current_A,
        ambient,
        _TRUTH["r0_ohm"],
        _TRUTH["r1_ohm"],
        _TRUTH["tau_s"],
    )
    temperature = euler_thermal_rollout(
        ambient,
        heat,
        ambient,
        dt_s,
        1.0 / _TRUTH["thermal_capacity_J_per_K"],
        _TRUTH["heat_transfer_W_per_K"] / _TRUTH["thermal_capacity_J_per_K"],
    )
    voltage = voltage + rng.normal(0.0, float(benchmark["voltage_noise_std_V"]), size=sample_count)
    temperature = temperature + rng.normal(
        0.0, float(benchmark["temperature_noise_std_K"]), size=sample_count
    )
    return pd.DataFrame(
        {
            "time_s": time_s,
            "current_A": current_A,
            "ambient_temperature_K": np.full(sample_count, ambient),
            "voltage_V": voltage,
            "temperature_K": temperature,
            "protocol_id": _protocol_id(c_rate, ambient),
        }
    )


def _score(train: pd.DataFrame, test: pd.DataFrame, benchmark: dict[str, Any]) -> dict[str, Any]:
    tau_grid = np.linspace(
        float(benchmark["tau_min_s"]),
        float(benchmark["tau_max_s"]),
        int(benchmark["tau_count"]),
    )
    alpha = float(benchmark["ridge_alpha"])
    dt_s = float(benchmark["sample_dt_s"])
    grey = GreyBoxTheveninSurrogate().fit(train, tau_grid)
    thermal = LumpedThermalSurrogate().fit(_thermal_segments(train, grey), dt_s)
    black_voltage = ProtocolRidgeSurrogate(alpha=alpha).fit(train, "voltage_V")
    black_temperature = ProtocolRidgeSurrogate(alpha=alpha).fit(train, "temperature_K")

    physics_voltage = {
        "train": _error_block(train["voltage_V"], grey.predict_voltage(train)),
        "test": _error_block(test["voltage_V"], grey.predict_voltage(test)),
    }
    physics_temperature = {
        "train": _error_block(
            train["temperature_K"], _aligned_temperature(train, grey, thermal, dt_s)
        ),
        "test": _error_block(
            test["temperature_K"], _aligned_temperature(test, grey, thermal, dt_s)
        ),
    }
    black_voltage_metrics = {
        "train": _error_block(train["voltage_V"], black_voltage.predict(train)),
        "test": _error_block(test["voltage_V"], black_voltage.predict(test)),
    }
    black_temperature_metrics = {
        "train": _error_block(train["temperature_K"], black_temperature.predict(train)),
        "test": _error_block(test["temperature_K"], black_temperature.predict(test)),
    }
    physics_test = physics_voltage["test"]["rmse"]
    black_test = black_voltage_metrics["test"]["rmse"]
    ratio = float(black_test / physics_test) if physics_test > 0 else float("inf")
    return {
        "models": {
            "black_box_ridge": {
                "voltage_V": black_voltage_metrics,
                "temperature_K": black_temperature_metrics,
                "physics": _slope_violations(black_voltage.predict, test),
            },
            "physics_informed": {
                "voltage_V": physics_voltage,
                "temperature_K": physics_temperature,
                "parameters": {
                    "r0_ohm": grey.r0_ohm,
                    "r1_ohm": grey.r1_ohm,
                    "tau_s": grey.tau_s,
                    "ocv_coefficients": [float(value) for value in grey.ocv_coefficients],
                    "ocv_derivative_min": ocv_derivative_min(grey.ocv_coefficients),
                    "thermal_capacity_J_per_K": thermal.thermal_capacity_J_per_K,
                    "heat_transfer_W_per_K": thermal.heat_transfer_W_per_K,
                    "resistances_nonnegative": grey.resistances_nonnegative,
                    "ocv_monotone_on_unit_interval": grey.ocv_monotone_on_unit_interval,
                },
                "physics": _slope_violations(grey.predict_voltage, test),
            },
        },
        "comparison": {
            "test_voltage_rmse_ratio_black_box_over_physics": ratio,
            "physics_informed_wins_voltage_extrapolation": bool(physics_test < black_test),
        },
    }


def _thermal_segments(frame: pd.DataFrame, grey: GreyBoxTheveninSurrogate) -> list[dict[str, Any]]:
    return [segment for _, segment in _indexed_segments(frame, grey)]


def _aligned_temperature(frame, grey, thermal: LumpedThermalSurrogate, dt_s: float) -> np.ndarray:
    predicted = np.empty(len(frame), dtype=float)
    for positions, segment in _indexed_segments(frame, grey):
        predicted[positions] = thermal.predict_temperature([segment], dt_s)
    return predicted


def _indexed_segments(frame: pd.DataFrame, grey: GreyBoxTheveninSurrogate):
    heat = np.asarray(grey.predict_heat_W(frame), dtype=float)
    segments = []
    for _, group in frame.groupby("protocol_id", sort=False):
        positions = frame.index.get_indexer(group.index)
        segments.append(
            (
                positions,
                {
                    "temperature_K": group["temperature_K"].to_numpy(dtype=float),
                    "heat_W": heat[positions],
                    "ambient_temperature_K": float(group["ambient_temperature_K"].iloc[0]),
                },
            )
        )
    return segments


def _error_block(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "max_abs_error": max_abs_error(y_true, y_pred),
    }


def _slope_violations(predict_voltage, frame: pd.DataFrame) -> dict[str, float]:
    base = np.asarray(predict_voltage(frame), dtype=float)
    current_frame = frame.copy()
    current_frame["current_A"] = frame["current_A"].to_numpy(dtype=float) + 0.1
    time_frame = frame.copy()
    time_frame["time_s"] = frame["time_s"].to_numpy(dtype=float) + 1.0
    dv_di = (np.asarray(predict_voltage(current_frame), dtype=float) - base) / 0.1
    dv_dt = (np.asarray(predict_voltage(time_frame), dtype=float) - base) / 1.0
    return {
        "ohmic_sign_violation_fraction": float(np.mean(dv_di > 1e-5)),
        "discharge_slope_violation_fraction": float(np.mean(dv_dt > 1e-5)),
    }


def _threshold_failures(report: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    physics = report["models"]["physics_informed"]
    black = report["models"]["black_box_ridge"]
    failures = []
    voltage_rmse = physics["voltage_V"]["test"]["rmse"]
    thermal_rmse = physics["temperature_K"]["test"]["rmse"]
    ratio = report["comparison"]["test_voltage_rmse_ratio_black_box_over_physics"]
    if voltage_rmse > float(thresholds["physics_test_voltage_rmse_V"]):
        failures.append(
            f"physics test voltage RMSE {voltage_rmse:.6g} V exceeds "
            f"{thresholds['physics_test_voltage_rmse_V']} V"
        )
    if thermal_rmse > float(thresholds["thermal_rollout_rmse_K"]):
        failures.append(
            f"physics thermal rollout RMSE {thermal_rmse:.6g} K exceeds "
            f"{thresholds['thermal_rollout_rmse_K']} K"
        )
    if ratio < float(thresholds["minimum_rmse_ratio_black_box_over_physics"]):
        failures.append(
            "black-box test voltage RMSE "
            f"{black['voltage_V']['test']['rmse']:.6g} V is not "
            f"{thresholds['minimum_rmse_ratio_black_box_over_physics']}x the physics model"
        )
    if not physics["parameters"]["resistances_nonnegative"]:
        failures.append("identified resistances are negative")
    if not physics["parameters"]["ocv_monotone_on_unit_interval"]:
        failures.append("identified open-circuit voltage is not monotone on [0, 1]")
    if physics["physics"]["ohmic_sign_violation_fraction"] > 0.0:
        failures.append("physics-informed voltage rises with current on a test sample")
    return failures


def _plain_thresholds(thresholds: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in thresholds.items()}


def _protocol_table(protocols: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    rows = []
    for protocol in protocols:
        c_rate = float(protocol["c_rate"])
        ambient = float(protocol["ambient_temperature_K"])
        rows.append(
            {
                "protocol_id": _protocol_id(c_rate, ambient),
                "c_rate": c_rate,
                "ambient_temperature_K": ambient,
            }
        )
    return rows


def _protocol_id(c_rate: float, ambient_temperature_K: float) -> str:
    return f"C{c_rate:.3f}_T{ambient_temperature_K:.2f}"
