from pathlib import Path

import pandas as pd

from battery_lab.benchmark import run_benchmark
from battery_lab.schema import load_yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "surrogate_benchmark.yaml"


def test_benchmark_is_reproducible_and_beats_the_ridge_baseline(tmp_path):
    spec = load_yaml(SPEC)
    first = run_benchmark(spec, tmp_path / "a")
    second = run_benchmark(spec, tmp_path / "b")
    report_a = (tmp_path / "a" / "benchmark_report.json").read_text(encoding="utf-8")
    report_b = (tmp_path / "b" / "benchmark_report.json").read_text(encoding="utf-8")
    assert report_a == report_b
    assert first["passed"] is True
    assert first["dataset"]["train_sha256"] == second["dataset"]["train_sha256"]
    assert first["metrics_sha256"] == second["metrics_sha256"]
    assert first["assumptions"]["comsol_used"] is False
    physics = first["models"]["physics_informed"]
    black = first["models"]["black_box_ridge"]
    assert physics["voltage_V"]["test"]["rmse"] < 0.01
    assert physics["temperature_K"]["test"]["rmse"] < 0.25
    assert first["comparison"]["test_voltage_rmse_ratio_black_box_over_physics"] > 10
    assert physics["physics"]["ohmic_sign_violation_fraction"] == 0.0
    assert physics["parameters"]["resistances_nonnegative"] is True
    assert physics["parameters"]["ocv_monotone_on_unit_interval"] is True
    assert black["voltage_V"]["test"]["rmse"] > physics["voltage_V"]["test"]["rmse"]
    train = pd.read_csv(tmp_path / "a" / "train.csv")
    test = pd.read_csv(tmp_path / "a" / "test.csv")
    assert set(train["protocol_id"]).isdisjoint(test["protocol_id"])
