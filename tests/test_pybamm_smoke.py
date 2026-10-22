import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from battery_lab.data_io import validate_timeseries

pybamm = pytest.importorskip("pybamm")
runner = importlib.import_module("battery_lab.pybamm_runner")
build_model = runner.build_model


def test_oregan_dfn_lumped_processes():
    model = build_model({"model": "DFN", "thermal": "lumped"})
    params = pybamm.ParameterValues("ORegan2022")
    params.process_model(model)


def test_x_lumped_selects_pouch_geometry():
    model = build_model({"model": "SPM", "thermal": "x-lumped"})
    assert model.options["cell geometry"] == "pouch"


def test_isothermal_temperature_falls_back_to_ambient():
    class EmptySolution:
        t = np.linspace(0.0, 1.0, 4)
        termination = "final time"

        def __getitem__(self, name):
            raise KeyError(name)

    values, warning = runner._temperature_series(
        EmptySolution(), np.linspace(0, 1, 4), 298.15, "isothermal"
    )
    assert warning
    assert np.all(values == 298.15)
    stopped = SimpleNamespace(t=np.array([0.0, 10.0]), termination="event: Minimum voltage [V]")
    t_out, early = runner._output_time(stopped, 100.0, 5)
    assert early
    assert t_out[-1] == pytest.approx(10.0)
    assert len(t_out) == 5


def test_short_spm_solve_honors_the_output_contract(tmp_path):
    config = {
        "project_id": "battery-ai-comsol-lab",
        "chemistry": "LFP_graphite",
        "cell_format": "generic",
        "engine": "pybamm",
        "model": "SPM",
        "thermal": "isothermal",
        "parameter_set": "Prada2013",
        "protocol_id": "unit_smoke",
        "c_rate": 1.0,
        "ambient_temperature_K": 298.15,
        "initial_soc": 1.0,
        "max_duration_s": 30,
        "output_points": 8,
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    csv_path, manifest_path = runner.run(config_path, tmp_path / "out")
    frame = pd.read_csv(csv_path)
    assert len(frame) == 8
    assert validate_timeseries(frame) == []
    assert frame["voltage_V"].iloc[-1] < frame["voltage_V"].iloc[0]
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "battery-ai-comsol-lab" in manifest
    assert "Prada2013" in manifest


def test_runner_helpers_cover_missing_series_and_bad_configs(tmp_path, monkeypatch):
    class BrokenVariable:
        def __call__(self, _time):
            raise TypeError("not callable in this build")

        data = np.array([3.4, 3.3, 3.2])

    class SampleSolution:
        t = np.array([0.0, 1.0, 2.0])

        def __getitem__(self, name):
            if name == "Voltage [V]":
                return BrokenVariable()
            raise KeyError(name)

    sampled = runner._sample_variable(SampleSolution(), "Voltage [V]", np.array([0.0, 1.0, 2.0]))
    assert sampled is not None
    assert runner._sample_variable(SampleSolution(), "missing", np.array([0.0, 1.0])) is None

    class NonFinite:
        t = np.array([0.0, 1.0])

        def __getitem__(self, name):
            return self

        def __call__(self, _time):
            return np.array([1.0, np.nan])

    assert runner._sample_variable(NonFinite(), "Voltage [V]", np.array([0.0, 1.0])) is None
    with pytest.raises(RuntimeError, match="finite cell temperature"):
        runner._temperature_series(SampleSolution(), np.array([0.0, 1.0]), 298.15, "lumped")

    class Params(dict):
        def update(self, values):
            super().update(values)

    with pytest.raises(KeyError, match="Current function"):
        runner._update_existing(Params(), {"Ambient temperature [K]": 298.15})
    warnings = runner._update_existing(
        Params({"Current function [A]": 1.0}),
        {"Current function [A]": 2.0, "Ambient temperature [K]": 1.0},
    )
    assert warnings

    config = tmp_path / "bad.yaml"
    config.write_text("chemistry: only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid configuration"):
        runner.run(config, tmp_path / "out")

    def explode(*_args, **_kwargs):
        raise OSError("no git")

    monkeypatch.setattr(runner.subprocess, "check_output", explode)
    assert runner._git_sha() is None

    class NoPayload:
        def __call__(self, _time):
            raise TypeError("unavailable")

    class Mismatched(NoPayload):
        data = np.array([1.0, 2.0, 3.0])

    class EntriesOnly(NoPayload):
        data = None
        entries = np.array([3.4, 3.3])

    class PayloadSolution:
        t = np.array([0.0, 1.0])

        def __init__(self, variable):
            self.variable = variable

        def __getitem__(self, _name):
            return self.variable

    times = np.array([0.0, 1.0])
    assert runner._sample_variable(PayloadSolution(NoPayload()), "Voltage [V]", times) is None
    assert runner._sample_variable(PayloadSolution(Mismatched()), "Voltage [V]", times) is None
    recovered = runner._sample_variable(PayloadSolution(EntriesOnly()), "Voltage [V]", times)
    assert recovered is not None


def _solver_config(path):
    import yaml

    path.write_text(
        yaml.safe_dump(
            {
                "chemistry": "LFP_graphite",
                "engine": "pybamm",
                "model": "SPM",
                "parameter_set": "Fake",
                "protocol_id": "fake",
                "c_rate": 1.0,
                "ambient_temperature_K": 298.15,
                "initial_soc": 1.0,
                "max_duration_s": 10,
                "output_points": 4,
                "thermal": "isothermal",
            }
        ),
        encoding="utf-8",
    )


def test_runner_rejects_unusable_solver_results(tmp_path, monkeypatch):
    class EmptyParams(dict):
        def __init__(self, _name):
            super().__init__()

        def update(self, values):
            super().update(values)

    config = tmp_path / "config.yaml"
    _solver_config(config)
    monkeypatch.setattr(runner.pybamm, "ParameterValues", EmptyParams)
    with pytest.raises(KeyError, match="Nominal cell capacity"):
        runner.run(config, tmp_path / "missing-capacity")

    class Params(dict):
        def __init__(self, _name):
            super().__init__(
                {
                    "Nominal cell capacity [A.h]": 5.0,
                    "Current function [A]": 1.0,
                }
            )

        def update(self, values):
            super().update(values)

    class Silent:
        t = np.linspace(0.0, 1.0, 4)
        termination = "final time"

        def __getitem__(self, _name):
            raise KeyError(_name)

    class HighVoltage(Silent):
        def __getitem__(self, _name):
            return self

        def __call__(self, time):
            return np.full(np.shape(time), 12.0)

    class Simulation:
        solution = Silent()

        def __init__(self, model, parameter_values):
            del model, parameter_values

        def solve(self, t_eval, initial_soc):
            del t_eval, initial_soc
            return self.solution

    monkeypatch.setattr(runner, "build_model", lambda _config: object())
    monkeypatch.setattr(runner.pybamm, "ParameterValues", Params)
    monkeypatch.setattr(runner.pybamm, "Simulation", Simulation)
    with pytest.raises(RuntimeError, match="finite voltage"):
        runner.run(config, tmp_path / "no-voltage")

    Simulation.solution = HighVoltage()
    with pytest.raises(RuntimeError, match="data contract"):
        runner.run(config, tmp_path / "bad-voltage")


def test_discharge_past_the_cutoff_stops_early_and_resamples(tmp_path):
    config = {
        "project_id": "battery-ai-comsol-lab",
        "chemistry": "LFP_graphite",
        "engine": "pybamm",
        "model": "SPM",
        "thermal": "isothermal",
        "parameter_set": "Prada2013",
        "protocol_id": "cutoff",
        "c_rate": 1.0,
        "ambient_temperature_K": 298.15,
        "initial_soc": 1.0,
        "max_duration_s": 10000,
        "output_points": 11,
    }
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    csv_path, manifest_path = runner.run(config_path, tmp_path / "out")
    frame = pd.read_csv(csv_path)
    assert len(frame) == 11
    assert frame["time_s"].iloc[-1] < 10000
    assert "before max_duration_s" in manifest_path.read_text(encoding="utf-8")
