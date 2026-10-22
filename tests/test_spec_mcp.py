import io
import json
from pathlib import Path

import pytest

from battery_lab.mcp import handle_message, read_message, serve, write_message
from battery_lab.spec import load_spec, run_spec_file, validate_spec

ROOT = Path(__file__).resolve().parents[1]


def test_repository_specs_validate_and_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in sorted((ROOT / "specs").glob("*.yaml")):
        assert validate_spec(load_spec(path), path) == [], path
    result = run_spec_file(ROOT / "specs" / "lfp_short_discharge.yaml", dry_run=True)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["resolved_config"]["model"] == "SPM"
    assert result["resolved_config"]["max_duration_s"] == 60
    leaked = {
        "api_version": "battery-lab/spec/v1",
        "kind": "surrogate-benchmark",
        "name": "leak",
        "output_dir": "runs/leak",
        "seed": 1,
        "benchmark": load_spec(ROOT / "specs" / "surrogate_benchmark.yaml")["benchmark"],
    }
    leaked["benchmark"]["test_protocols"] = list(leaked["benchmark"]["train_protocols"])
    assert any("repeats" in error for error in validate_spec(leaked))
    benchmark = run_spec_file(
        ROOT / "specs" / "surrogate_benchmark.yaml",
        output_dir=tmp_path / "bench",
    )
    assert benchmark["ok"] is True
    assert benchmark["metrics"]["passed"] is True
    assert Path(benchmark["artifacts"]["benchmark_report_json"]).is_file()


def test_invalid_override_and_missing_pybamm_are_reported(monkeypatch, tmp_path):
    spec = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "bad-override",
        "output_dir": "runs/bad",
        "base_config": str(ROOT / "configs" / "lfp_graphite_baseline.yaml"),
        "overrides": {"crate": 2},
    }
    errors = validate_spec(spec)
    assert any("unknown override" in error for error in errors)

    inline = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "inline-spm",
        "output_dir": "runs/inline",
        "simulation": {
            "chemistry": "LFP_graphite",
            "engine": "pybamm",
            "model": "SPM",
            "parameter_set": "Prada2013",
            "protocol_id": "inline",
            "c_rate": float("nan"),
        },
    }
    assert any("c_rate" in error for error in validate_spec(inline))

    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    missing = {
        "api_version": "battery-lab/spec/v1",
        "kind": "simulation",
        "name": "needs-pybamm",
        "output_dir": str(tmp_path / "out"),
        "base_config": str(ROOT / "configs" / "lfp_graphite_baseline.yaml"),
        "overrides": {"max_duration_s": 30, "output_points": 4, "model": "SPM"},
    }
    from battery_lab.spec import run_spec

    result = run_spec(missing, output_dir=tmp_path / "out", dry_run=False)
    assert result["ok"] is False
    assert any("pybamm" in error for error in result["errors"])


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def test_mcp_stdio_round_trip(tmp_path):
    spec = (ROOT / "specs" / "surrogate_benchmark.yaml").read_text(encoding="utf-8")
    calls = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "run_spec",
                "arguments": {"yaml": spec, "dry_run": True, "output_dir": str(tmp_path / "dry")},
            },
        },
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "doctor"}},
    ]
    stdout = io.BytesIO()
    serve(io.BytesIO(b"".join(_frame(call) for call in calls)), stdout)
    stdout.seek(0)
    messages = []
    while True:
        message = read_message(stdout)
        if message is None:
            break
        messages.append(message)
    assert [message["id"] for message in messages] == [1, 2, 3, 4, 5]
    assert messages[0]["result"]["serverInfo"]["name"] == "battery-lab"
    tool_names = [tool["name"] for tool in messages[1]["result"]["tools"]]
    assert "run_spec" in tool_names
    dry = json.loads(messages[3]["result"]["content"][0]["text"])
    assert dry["ok"] is True
    assert dry["dry_run"] is True
    doctor = json.loads(messages[4]["result"]["content"][0]["text"])
    assert "packages" in doctor


def test_mcp_unknown_method_and_tool():
    unknown = handle_message({"jsonrpc": "2.0", "id": 7, "method": "nope"})
    assert unknown["error"]["code"] == -32601
    missing_tool = handle_message(
        {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "missing"}}
    )
    assert missing_tool["result"]["isError"] is True
    buffer = io.BytesIO()
    write_message(buffer, {"ok": True})
    buffer.seek(0)
    assert read_message(buffer) == {"ok": True}


def test_sweep_validation_does_not_import_pybamm(tmp_path):
    bad = tmp_path / "sweep.yaml"
    bad.write_text(
        "base_config: missing.yaml\nc_rates: [.nan]\nambient_temperatures_K: [298.15]\n",
        encoding="utf-8",
    )
    from battery_lab.sweep import run_sweep

    with pytest.raises(ValueError, match="c_rates"):
        run_sweep(bad)
