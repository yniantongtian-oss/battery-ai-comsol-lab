import copy
import io
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from battery_lab.cli import (
    _cmd_benchmark,
    _cmd_doctor,
    _cmd_run_spec,
    _cmd_validate_data,
    doctor_report,
)
from battery_lab.config import (
    validate_pack_config,
    validate_simulation_config,
    validate_sweep_config,
)
from battery_lab.cross_validation import compare_result_tables
from battery_lab.mcp import _call_tool, handle_message, main, read_message
from battery_lab.schema import load_yaml
from battery_lab.spec import (
    _run_simulation_spec,
    load_spec_text,
    resolve_simulation_config,
    run_spec,
    run_spec_file,
    validate_spec,
    validate_spec_file,
)
from battery_lab.target_cell import evidence_readiness, validate_target_cell_spec

ROOT = Path(__file__).resolve().parents[1]


def _benchmark():
    return {
        "api_version": "battery-lab/spec/v1",
        "kind": "surrogate-benchmark",
        "name": "bench",
        "output_dir": "runs/bench",
        "seed": 1,
        "benchmark": {
            "sample_dt_s": 10,
            "soc_start": 1.0,
            "soc_stop": 0.05,
            "voltage_noise_std_V": 0.0,
            "temperature_noise_std_K": 0.0,
            "ridge_alpha": 0.0,
            "tau_min_s": 20,
            "tau_max_s": 90,
            "tau_count": 3,
            "train_protocols": [{"c_rate": 1.0, "ambient_temperature_K": 298.15}],
            "test_protocols": [{"c_rate": 2.0, "ambient_temperature_K": 298.15}],
            "thresholds": {
                "physics_test_voltage_rmse_V": 0.01,
                "thermal_rollout_rmse_K": 0.25,
                "minimum_rmse_ratio_black_box_over_physics": 10,
            },
        },
    }


def _errors(spec):
    return " ".join(validate_spec(spec))


def test_benchmark_spec_rejects_each_invalid_field():
    wrong_api = _benchmark()
    wrong_api["api_version"] = "other"
    assert "api_version" in _errors(wrong_api)

    wrong_kind = _benchmark()
    wrong_kind["kind"] = "pack"
    assert "kind" in _errors(wrong_kind)

    blank_dir = _benchmark()
    blank_dir["output_dir"] = "  "
    assert "output_dir" in _errors(blank_dir)

    bad_seed = _benchmark()
    bad_seed["seed"] = True
    assert "seed" in _errors(bad_seed)
    huge_seed = _benchmark()
    huge_seed["seed"] = 2**32
    assert "seed" in _errors(huge_seed)

    not_mapping = _benchmark()
    not_mapping["benchmark"] = "nope"
    assert "benchmark must be a mapping" in _errors(not_mapping)

    unknown = _benchmark()
    unknown["benchmark"]["extra"] = 1
    assert "unknown benchmark fields" in _errors(unknown)

    missing = _benchmark()
    missing["benchmark"] = {"sample_dt_s": 1}
    assert "missing benchmark fields" in _errors(missing)

    bad_dt = _benchmark()
    bad_dt["benchmark"]["sample_dt_s"] = 0
    assert "sample_dt_s" in _errors(bad_dt)
    bad_dt["benchmark"]["sample_dt_s"] = "slow"
    assert "sample_dt_s" in _errors(bad_dt)

    inverted = _benchmark()
    inverted["benchmark"]["tau_max_s"] = 1
    inverted["benchmark"]["tau_min_s"] = "late"
    assert "tau_min_s" in _errors(inverted)
    inverted["benchmark"]["tau_min_s"] = 20
    assert "tau_max_s" in _errors(inverted)

    tau_count = _benchmark()
    tau_count["benchmark"]["tau_count"] = "many"
    assert "tau_count" in _errors(tau_count)
    tau_count["benchmark"]["tau_count"] = 1
    assert "tau_count" in _errors(tau_count)

    soc = _benchmark()
    soc["benchmark"]["soc_start"] = "full"
    assert "soc_start" in _errors(soc)
    soc["benchmark"]["soc_start"] = 0.2
    soc["benchmark"]["soc_stop"] = 0.5
    assert "soc_stop" in _errors(soc)

    noise = _benchmark()
    noise["benchmark"]["voltage_noise_std_V"] = -0.1
    noise["benchmark"]["temperature_noise_std_K"] = "loud"
    noise["benchmark"]["ridge_alpha"] = -1
    text = _errors(noise)
    assert "voltage_noise_std_V" in text
    assert "temperature_noise_std_K" in text
    assert "ridge_alpha" in text

    protocols = _benchmark()
    protocols["benchmark"]["train_protocols"] = "cc"
    protocols["benchmark"]["test_protocols"] = []
    assert "train_protocols" in _errors(protocols)

    duplicate = _benchmark()
    duplicate["benchmark"]["train_protocols"] = [
        {"c_rate": 1, "ambient_temperature_K": 298.15, "note": "x"},
        "cc",
        {"c_rate": 0, "ambient_temperature_K": 10},
        {"c_rate": "fast", "ambient_temperature_K": "cold"},
        {"c_rate": 1, "ambient_temperature_K": 298.15},
    ]
    text = _errors(duplicate)
    assert "unknown fields" in text
    assert "entries must be mappings" in text
    assert "c_rate must be > 0" in text
    assert "ambient_temperature_K must be between" in text
    assert "duplicate protocol" in text

    overlap = _benchmark()
    overlap["benchmark"]["test_protocols"] = [{"c_rate": 1.0, "ambient_temperature_K": 298.15}]
    assert "repeats a training protocol" in _errors(overlap)

    short = _benchmark()
    short["benchmark"]["sample_dt_s"] = 1000
    short["benchmark"]["train_protocols"] = [
        {"ambient_temperature_K": 298.15},
        {"c_rate": 0, "ambient_temperature_K": 298.15},
        {"c_rate": "fast", "ambient_temperature_K": 298.15},
        {"c_rate": 100, "ambient_temperature_K": 298.15},
    ]
    short["benchmark"]["test_protocols"] = [{"c_rate": 1.0, "ambient_temperature_K": 298.15}]
    assert "fewer than 5 samples" in _errors(short)

    thresholds = _benchmark()
    thresholds["benchmark"]["thresholds"] = []
    assert "thresholds must be a mapping" in _errors(thresholds)
    thresholds["benchmark"]["thresholds"] = {"physics_test_voltage_rmse_V": 0}
    assert "missing thresholds" in _errors(thresholds)
    thresholds["benchmark"]["thresholds"] = {
        "physics_test_voltage_rmse_V": 0,
        "thermal_rollout_rmse_K": "hot",
        "minimum_rmse_ratio_black_box_over_physics": 10,
    }
    text = _errors(thresholds)
    assert "must be > 0" in text
    assert "finite number" in text

    with pytest.raises(ValueError, match="200000"):
        load_spec_text("x" * 200_001)
    assert validate_spec_file(ROOT / "specs" / "surrogate_benchmark.yaml") == []


def test_simulation_spec_resolution_reports_structural_errors(tmp_path):
    base = ROOT / "configs" / "lfp_graphite_baseline.yaml"
    both = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "both",
        "output_dir": "runs/both",
        "base_config": str(base),
        "simulation": {},
    }
    assert "exactly one" in _errors(both)
    resolved, errors = resolve_simulation_config(both)
    assert resolved is None and errors

    overrides = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "overrides",
        "output_dir": "runs/overrides",
        "base_config": str(base),
        "overrides": {"not_a_field": 1, "model_options": ["bad"], "c_rate": True},
    }
    assert "unknown override" in _errors(overrides)
    overrides["overrides"] = {"model_options": {"flag": [1]}}
    assert "model_options" in _errors(overrides)
    overrides["overrides"] = {"model_options": {"operating mode": "current"}}
    assert validate_spec(overrides) == []
    overrides["overrides"] = None
    assert validate_spec(overrides) == []

    inline = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "inline",
        "output_dir": "runs/inline",
        "simulation": [],
        "overrides": [],
    }
    assert "overrides must be a mapping" in _errors(inline)
    inline["overrides"] = {"not_a_field": 1}
    inline["simulation"] = {"engine": "pybamm"}
    assert "unknown override" in _errors(inline)
    inline["overrides"] = {}
    assert "simulation must be a mapping" not in _errors(inline)
    inline["simulation"] = []
    assert "simulation must be a mapping" in _errors(inline)

    inline = load_yaml(base)
    inline_spec = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "inline-ok",
        "output_dir": "runs/inline-ok",
        "simulation": inline,
    }
    resolved, resolve_errors = resolve_simulation_config(inline_spec)
    assert resolve_errors == []
    assert resolved["parameter_set"] == "Prada2013"
    failed_resolve = _run_simulation_spec(
        {
            "name": "missing-base",
            "output_dir": "runs/missing-base",
            "base_config": "does-not-exist.yaml",
        },
        spec_path=None,
        dry_run=True,
    )
    assert failed_resolve["ok"] is False

    missing = run_spec_file(tmp_path / "absent.yaml")
    assert missing["ok"] is False
    bad_output = run_spec(overrides, output_dir=" \x00 ")
    assert bad_output["ok"] is False

    def explode(_spec, _output):
        raise ValueError("benchmark failed")

    spec = _benchmark()
    spec["output_dir"] = "runs/boom"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("battery_lab.benchmark.run_benchmark", explode)
    try:
        failed = run_spec(spec)
    finally:
        monkeypatch.undo()
    assert failed["ok"] is False
    assert "benchmark failed" in failed["errors"][0]


def test_config_ranges_reject_unsupported_values(tmp_path):
    errors = validate_simulation_config(
        {
            "chemistry": "LFP",
            "engine": "comsol",
            "model": "XYZ",
            "thermal": "nope",
            "parameter_set": "P",
            "protocol_id": "cc",
            "c_rate": -1,
            "ambient_temperature_K": 10,
            "initial_soc": 2,
            "max_duration_s": -5,
            "output_points": 1,
        }
    )
    assert any("engine" in error for error in errors)
    assert any("model" in error for error in errors)
    assert any("thermal" in error for error in errors)
    assert any("output_points" in error for error in errors)

    assert validate_sweep_config({})
    zero_rate = validate_sweep_config({"c_rates": [0], "base_config": "x"})
    assert any("c_rates" in error for error in zero_rate)
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text("base_config: missing.yaml\n", encoding="utf-8")
    missing_base = validate_sweep_config(
        {
            "base_config": "missing.yaml",
            "c_rates": [1],
            "ambient_temperatures_K": [10],
        },
        sweep,
    )
    assert any("does not exist" in error for error in missing_base)

    pack = tmp_path / "pack.yaml"
    pack.write_text("cell_config: missing.yaml\n", encoding="utf-8")
    errors = validate_pack_config(
        {
            "cell_config": "missing.yaml",
            "series_cells": 0,
            "parallel_cells": True,
            "initial_soc_mean": 2,
            "initial_soc_std": -1,
        },
        pack,
    )
    assert errors
    errors = validate_pack_config(
        {"series_cells": 1, "parallel_cells": 1, "initial_soc_mean": float("nan")},
        None,
    )
    assert any("initial_soc_mean" in error for error in errors)


def test_target_cell_evidence_must_point_at_real_files(tmp_path):
    spec = {
        "target_id": "cell",
        "cell_revision": "a",
        "chemistry": "LFP",
        "form_factor": "pouch",
        "parameter_set": "Prada2013",
        "reference_simulation": "missing.yaml",
        "evidence": {
            "datasheet": {"status": "weird"},
            "capacity_reference": {"status": "available"},
            "ocv_soc": {"status": "validated", "path": "gone.csv", "sha256": "ab"},
            "hppc": {"status": "required"},
            "thermal": "nope",
        },
        "protocols": {"capacity_reference": ""},
        "replicates": {"minimum_per_condition": 0},
        "acceptance_thresholds": ["bad"],
    }
    errors = " ".join(validate_target_cell_spec(spec, root=tmp_path))
    assert "does not exist" in errors
    assert "status must be one of" in errors
    assert "path required" in errors
    assert "sha256" in errors
    assert "missing protocol" in errors
    assert "minimum_per_condition" in errors
    assert "acceptance_thresholds must be a mapping" in errors
    numeric = copy.deepcopy(spec)
    numeric["acceptance_thresholds"] = {"voltage_rmse_V": "nope"}
    assert any("numeric" in error for error in validate_target_cell_spec(numeric))
    assert evidence_readiness({"evidence": {}})["datasheet"] == "missing"


def test_cross_validation_requires_two_overlapping_samples():
    reference = pd.DataFrame({"time_s": [0.0, 10.0], "voltage_V": [1.0, 2.0]})
    candidate = pd.DataFrame({"time_s": [0.0, 0.1], "voltage_V": [1.0, 1.0]})
    with pytest.raises(ValueError, match="fewer than two"):
        compare_result_tables(reference, candidate, signals=["voltage_V"])


def test_cli_doctor_and_benchmark_cover_missing_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert _cmd_doctor(Namespace(json=False)) == 0
    assert "not found" in capsys.readouterr().out

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "bad.yaml").write_text("engine: comsol\n", encoding="utf-8")
    assert _cmd_doctor(Namespace(json=False)) == 1
    assert "FAIL config" in capsys.readouterr().out

    valid = tmp_path / "series.csv"
    valid.write_text(
        "time_s,current_A,voltage_V,temperature_K\n0,1,3.5,298.15\n",
        encoding="utf-8",
    )
    assert _cmd_validate_data(Namespace(path=str(valid))) == 0
    assert capsys.readouterr().out.strip() == "OK"

    missing_spec = Namespace(
        spec=str(tmp_path / "missing.yaml"), out=None, dry_run=False, json=False
    )
    assert _cmd_benchmark(missing_spec) == 1
    monkeypatch.chdir(ROOT)

    def hidden():
        report = doctor_report()
        for package in report["packages"]:
            if package["name"] == "pybamm":
                package["installed"] = False
                package["version"] = None
        return report

    monkeypatch.setattr("battery_lab.cli.doctor_report", hidden)
    assert _cmd_doctor(Namespace(json=False)) == 0
    assert "pybamm" in capsys.readouterr().out

    def missing_solver(*_args, **_kwargs):
        return {
            "ok": False,
            "kind": "simulation",
            "name": "x",
            "errors": ["optional runtime dependency 'pybamm' is not installed"],
            "artifacts": {},
            "metrics": {},
        }

    monkeypatch.setattr("battery_lab.spec.run_spec_file", missing_solver)
    assert _cmd_run_spec(Namespace(spec="x", out=None, dry_run=False, json=False)) == 2


def test_mcp_covers_argument_and_path_branches(tmp_path, monkeypatch):
    assert read_message(io.BytesIO(b" \r\nContent-Length: 2\r\n\r\n{}")) is None
    noted = handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert noted is None
    initialized = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        }
    )
    assert initialized["result"]["protocolVersion"] == "2024-11-05"

    valid = (ROOT / "specs" / "surrogate_benchmark.yaml").read_text(encoding="utf-8")
    assert _call_tool("validate_spec", {})["ok"] is False
    none_args = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "doctor", "arguments": None},
        }
    )
    assert none_args["result"]["isError"] is False
    checked = _call_tool("validate_spec", {"yaml": valid})
    assert checked["ok"] is True
    spec_path = str(ROOT / "specs" / "lfp_short_discharge.yaml")
    checked = _call_tool("validate_spec", {"path": spec_path})
    assert checked["ok"] is True
    ran = _call_tool(
        "run_spec",
        {"path": str(ROOT / "specs" / "surrogate_benchmark.yaml"), "dry_run": True},
    )
    assert ran["ok"] is True
    assert _call_tool("run_spec", {"yaml": "a: 1\n", "path": "x"})["ok"] is False
    listed = _call_tool("list_configs", {"root": str(ROOT / "specs" / "surrogate_benchmark.yaml")})
    assert listed["ok"] is True
    assert listed["paths"]
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"")))
    main()
