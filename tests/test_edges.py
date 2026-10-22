import io
import json

import numpy as np
import pandas as pd
import pytest

from battery_lab.benchmark import _threshold_failures
from battery_lab.config import validate_pack_config, validate_sweep_config
from battery_lab.cross_validation import compare_result_tables
from battery_lab.data_io import read_timeseries, validate_timeseries
from battery_lab.ecm import TheveninParameters, TheveninState, coulomb_count, thevenin_step
from battery_lab.mcp import _call_tool, handle_message, read_message
from battery_lab.metrics import rmse
from battery_lab.schema import load_yaml
from battery_lab.spec import load_spec_text, run_spec, validate_spec
from battery_lab.sweep import run_sweep
from battery_lab.target_cell import validate_target_cell_spec


def test_validation_edges_reject_bad_records(tmp_path):
    assert validate_timeseries(pd.DataFrame({"time_s": [0.0]}))
    assert validate_timeseries(
        pd.DataFrame(
            {
                "time_s": [0.0],
                "current_A": [1.0],
                "voltage_V": [3.5],
                "temperature_K": [298.15],
            }
        )
    ) == []
    empty = pd.DataFrame(columns=["time_s", "current_A", "voltage_V", "temperature_K"])
    assert any("empty" in error for error in validate_timeseries(empty))
    cold = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "current_A": [1.0, 1.0],
            "voltage_V": [3.7, 3.6],
            "temperature_K": [100.0, 100.0],
        }
    )
    assert any("temperature_K" in error for error in validate_timeseries(cold))
    csv_path = tmp_path / "series.csv"
    cold.to_csv(csv_path, index=False)
    assert len(read_timeseries(csv_path)) == 2

    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        coulomb_count(0.5, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        thevenin_step(TheveninState(0.5), 1.0, -1.0, TheveninParameters(2, 0.01, 0.02, 10), 3.7)
    with pytest.raises(ValueError):
        thevenin_step(TheveninState(np.nan), 1.0, 1.0, TheveninParameters(2, 0.01, 0.02, 10), 3.7)
    with pytest.raises(ValueError):
        thevenin_step(TheveninState(0.5), 1.0, 1.0, TheveninParameters(2, -0.01, 0.02, 10), 3.7)

    assert validate_pack_config({"series_cells": True, "parallel_cells": 0})
    assert validate_sweep_config({"c_rates": [], "ambient_temperatures_K": [10, "x"]})
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(listing)
    with pytest.raises(ValueError):
        load_spec_text("[]")
    with pytest.raises(ValueError):
        load_spec_text("x" * 200_001)


def test_cross_validation_edge_cases():
    with pytest.raises(ValueError, match="time"):
        compare_result_tables(
            pd.DataFrame({"voltage_V": [1.0, 1.0]}),
            pd.DataFrame({"time_s": [0.0, 1.0], "voltage_V": [1.0, 1.0]}),
        )
    empty = pd.DataFrame({"time_s": [np.nan], "voltage_V": [1.0]})
    with pytest.raises(ValueError, match="finite"):
        compare_result_tables(empty, empty)
    duplicated = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0],
            "voltage_V": [1.0, 3.0, 2.0],
            "temperature_K": [298.0, 298.0, 299.0],
        }
    )
    report = compare_result_tables(duplicated, duplicated)
    assert report["signals"]["voltage_V"]["status"] == "ok"
    sparse = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "voltage_V": [np.nan, np.nan, 1.0]})
    dense = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "voltage_V": [1.0, 1.0, 1.0]})
    sparse_report = compare_result_tables(sparse, dense, signals=["voltage_V"])
    assert sparse_report["signals"]["voltage_V"]["status"] == "insufficient_finite_data"


def test_target_cell_and_spec_validation_report_specific_errors():
    errors = validate_target_cell_spec(
        {"replicates": "nope", "acceptance_thresholds": {"voltage_rmse_V": 0}}
    )
    assert errors
    spec = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "both",
        "output_dir": "/tmp/absolute",
        "base_config": "missing.yaml",
        "simulation": {"engine": "pybamm"},
        "overrides": {"model_options": {"nested": {"nope": 1}}, "c_rate": True},
    }
    assert validate_spec(spec)
    benchmark = {
        "api_version": "battery-lab/spec/v1",
        "kind": "surrogate-benchmark",
        "name": "bad name",
        "output_dir": "runs/bad",
        "seed": -1,
        "benchmark": {"sample_dt_s": 0},
        "extra": True,
    }
    assert validate_spec(benchmark)
    missing = run_spec(
        {
            "api_version": "battery-lab/spec/v1",
            "kind": "simulation",
            "name": "no-file",
            "output_dir": "runs/no-file",
            "base_config": "does-not-exist.yaml",
        }
    )
    assert missing["ok"] is False


def test_threshold_gate_lists_each_failure_mode():
    report = {
        "models": {
            "physics_informed": {
                "voltage_V": {"test": {"rmse": 1.0}},
                "temperature_K": {"test": {"rmse": 5.0}},
                "parameters": {
                    "resistances_nonnegative": False,
                    "ocv_monotone_on_unit_interval": False,
                },
                "physics": {"ohmic_sign_violation_fraction": 0.2},
            },
            "black_box_ridge": {"voltage_V": {"test": {"rmse": 1.1}}},
        },
        "comparison": {"test_voltage_rmse_ratio_black_box_over_physics": 1.1},
    }
    thresholds = {
        "physics_test_voltage_rmse_V": 0.01,
        "thermal_rollout_rmse_K": 0.25,
        "minimum_rmse_ratio_black_box_over_physics": 10,
    }
    assert len(_threshold_failures(report, thresholds)) == 6


def test_sweep_rejects_a_base_config_that_cannot_simulate(tmp_path):
    (tmp_path / "base.yaml").write_text("notes: incomplete\n", encoding="utf-8")
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        "base_config: base.yaml\nc_rates: [1]\nambient_temperatures_K: [298.15]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid sweep point"):
        run_sweep(sweep)


def test_mcp_argument_errors_and_directory_listing():
    assert read_message(io.BytesIO(b"\r\n")) is None
    assert read_message(io.BytesIO(b"Content-Length: 8\r\n\r\n{}")) is None
    invalid = handle_message({"id": 1})
    assert invalid["error"]["code"] == -32600
    bad_args = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "doctor", "arguments": []},
        }
    )
    assert bad_args["result"]["isError"] is True
    both = _call_tool("validate_spec", {"yaml": "a: 1\n", "path": "x"})
    assert both["ok"] is False
    assert _call_tool("run_spec", {})["ok"] is False
    listed = _call_tool("list_configs", {"root": "configs"})
    assert listed["ok"] is True
    assert listed["paths"]
    missing = _call_tool("list_configs", {"root": "not-a-directory"})
    assert missing["ok"] is False
    broken = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "validate_spec", "arguments": {"yaml": "[]"}},
        }
    )
    assert broken["result"]["isError"] is True
    text = json.loads(broken["result"]["content"][0]["text"])
    assert text["ok"] is False
