from pathlib import Path

import pandas as pd

from battery_lab.cross_validation import compare_result_tables
from battery_lab.target_cell import (
    load_target_cell_spec,
    validate_target_cell_spec,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase_b_target_definition_is_structurally_valid():
    path = ROOT / "configs" / "target_cell_lgm50_reference.yaml"
    spec = load_target_cell_spec(path)
    assert validate_target_cell_spec(spec, root=ROOT) == []


def test_cross_validation_returns_zero_for_identical_series():
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "voltage_V": [4.2, 4.1, 4.0, 3.9],
            "temperature_K": [298.15, 298.2, 298.3, 298.4],
        }
    )
    report = compare_result_tables(frame, frame)
    for signal in ("voltage_V", "temperature_K"):
        metrics = report["signals"][signal]
        assert metrics["status"] == "ok"
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["max_abs_error"] == 0.0


def test_cross_validation_marks_missing_signal_without_fabricating_values():
    reference = pd.DataFrame({"time_s": [0.0, 1.0], "voltage_V": [4.0, 3.9]})
    candidate = pd.DataFrame({"time_s": [0.0, 1.0], "voltage_V": [4.0, 3.9]})
    report = compare_result_tables(reference, candidate)
    assert report["signals"]["temperature_K"]["status"] == "missing"
