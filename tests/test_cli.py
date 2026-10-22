import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from battery_lab.cli import (
    _cmd_benchmark,
    _cmd_doctor,
    _cmd_mcp,
    _cmd_run_spec,
    _cmd_simulate,
    _cmd_sweep,
    _cmd_validate_configs,
    _cmd_validate_data,
    build_parser,
    doctor_report,
    main,
)


def test_cli_parser_exposes_doctor_and_validation():
    parser = build_parser()
    assert parser.prog == "battery-lab"


def test_all_repository_configs_validate():
    root = Path(__file__).resolve().parents[1]
    for folder in ("configs", "specs"):
        args = type("Args", (), {"path": str(root / folder)})()
        assert _cmd_validate_configs(args) == 0


def test_doctor_accepts_the_repository(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    report = doctor_report()
    assert report["ok"] is True
    assert report["config_errors"] == []
    assert report["version"]


def test_run_spec_dry_run_prints_resolved_model(capsys):
    spec = Path(__file__).resolve().parents[1] / "specs" / "lfp_short_discharge.yaml"
    code = _cmd_run_spec(Namespace(spec=str(spec), out=None, dry_run=True, json=True))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved_config"]["parameter_set"] == "Prada2013"
    assert payload["resolved_config"]["output_points"] == 12


def _fake_series(config_path, out_dir):
    del config_path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "timeseries.csv"
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "current_A": [1.0, 1.0],
            "voltage_V": [3.6, 3.4],
            "temperature_K": [298.15, 298.2],
        }
    ).to_csv(csv_path, index=False)
    manifest = out / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return csv_path, manifest


def test_cli_commands_report_success_and_failure(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("battery_lab.pybamm_runner.run", _fake_series)
    assert _cmd_simulate(Namespace(config="c", out=tmp_path / "sim", json=True)) == 0
    assert _cmd_simulate(Namespace(config="c", out=tmp_path / "sim2", json=False)) == 0

    def fail_run(*_args, **_kwargs):
        raise ValueError("bad config")

    monkeypatch.setattr("battery_lab.pybamm_runner.run", fail_run)
    assert _cmd_simulate(Namespace(config="c", out="o", json=True)) == 1
    assert _cmd_simulate(Namespace(config="c", out="o", json=False)) == 1

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert _cmd_simulate(Namespace(config="c", out="o", json=True)) == 2
    assert _cmd_simulate(Namespace(config="c", out="o", json=False)) == 2
    assert _cmd_sweep(Namespace(config="c", json=True)) == 2
    assert _cmd_sweep(Namespace(config="c", json=False)) == 2
    capsys.readouterr()

    root = Path(__file__).resolve().parents[1]
    spec = root / "specs" / "lfp_short_discharge.yaml"
    assert _cmd_benchmark(Namespace(spec=str(spec), out=None, dry_run=False, json=False)) == 1
    assert _cmd_benchmark(Namespace(spec=str(spec), out=None, dry_run=False, json=True)) == 1
    benchmark = root / "specs" / "surrogate_benchmark.yaml"
    assert _cmd_benchmark(Namespace(spec=str(benchmark), out=None, dry_run=True, json=False)) == 0

    missing = tmp_path / "missing.yaml"
    assert _cmd_validate_configs(Namespace(path=str(missing))) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _cmd_validate_configs(Namespace(path=str(empty))) == 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("chemistry: only\n", encoding="utf-8")
    assert _cmd_validate_configs(Namespace(path=str(bad))) == 1

    data = tmp_path / "bad.csv"
    data.write_text("time_s\n0\n", encoding="utf-8")
    assert _cmd_validate_data(Namespace(path=str(data))) == 1
    monkeypatch.chdir(root)
    assert _cmd_doctor(Namespace(json=True)) == 0
    assert _cmd_doctor(Namespace(json=False)) == 0
    monkeypatch.setattr("battery_lab.mcp.serve", lambda: None)
    assert _cmd_mcp(Namespace()) == 0
    monkeypatch.setattr("sys.argv", ["battery-lab", "doctor", "--json"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_sweep_and_spec_execution_use_the_runner(monkeypatch, capsys, tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "chemistry: LFP\nengine: pybamm\nmodel: SPM\nparameter_set: Prada2013\nprotocol_id: cc\n",
        encoding="utf-8",
    )
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        f"base_config: {base.name}\nc_rates: [0.5, 1]\nambient_temperatures_K: [298.15]\n"
        f"output_root: {tmp_path / 'sweep-out'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("battery_lab.pybamm_runner.run", _fake_series)
    assert _cmd_sweep(Namespace(config=str(sweep), json=True)) == 0
    assert _cmd_sweep(Namespace(config=str(sweep), json=False)) == 0

    def fail_sweep(*_args, **_kwargs):
        raise RuntimeError("sweep failed")

    monkeypatch.setattr("battery_lab.sweep.run_sweep", fail_sweep)
    assert _cmd_sweep(Namespace(config=str(sweep), json=True)) == 1
    assert _cmd_sweep(Namespace(config=str(sweep), json=False)) == 1

    spec = Path(__file__).resolve().parents[1] / "specs" / "lfp_short_discharge.yaml"
    monkeypatch.setattr("battery_lab.pybamm_runner.run", _fake_series)
    code = _cmd_run_spec(
        Namespace(spec=str(spec), out=tmp_path / "agent", dry_run=False, json=False)
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "timeseries_csv" in text

    def fail_spec(*_args, **_kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr("battery_lab.pybamm_runner.run", fail_spec)
    code = _cmd_run_spec(
        Namespace(spec=str(spec), out=tmp_path / "bad", dry_run=False, json=False)
    )
    assert code == 1
