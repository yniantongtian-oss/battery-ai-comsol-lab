import sys
import types

import pytest

from battery_lab.comsol_bridge import ComsolSession, ComsolUnavailableError


def test_comsol_session_reports_a_missing_import(monkeypatch):
    real_import = __import__

    def fail_mph(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mph" or name.startswith("mph."):
            raise ImportError("mph is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fail_mph)
    with pytest.raises(ComsolUnavailableError):
        ComsolSession()


def test_comsol_session_reports_a_missing_installation(monkeypatch):
    fake = types.ModuleType("mph")

    def start(**kwargs):
        raise RuntimeError("Could not find a Comsol installation")

    fake.start = start
    monkeypatch.setitem(sys.modules, "mph", fake)
    with pytest.raises(ComsolUnavailableError):
        ComsolSession()


def test_comsol_session_drives_a_fake_client_without_a_license(monkeypatch, tmp_path):
    model_calls = {}

    class FakeModel:
        def parameter(self, name, value):
            model_calls.setdefault("parameters", {})[name] = value

        def solve(self, study=None):
            model_calls["study"] = study

        def evaluate(self, expressions, unit=None):
            return {"expressions": expressions, "unit": unit}

        def save(self, path):
            model_calls["saved"] = path

    class FakeClient:
        def __init__(self):
            self.removed = False

        def load(self, path):
            model_calls["loaded"] = path
            return FakeModel()

        def remove(self, model):
            self.removed = True
            model_calls["removed_model"] = model

    client = FakeClient()
    fake = types.ModuleType("mph")
    fake.start = lambda **kwargs: client
    monkeypatch.setitem(sys.modules, "mph", fake)

    with pytest.raises(RuntimeError, match="No COMSOL model"):
        with ComsolSession(cores=1) as session:
            session.run_study("too-soon")

    with ComsolSession(cores=2) as session:
        session.open_model(tmp_path / "cell.mph")
        session.set_parameters({"T_amb": 298.15, "I_app": 5})
        session.run_study("discharge")
        assert session.evaluate("V")["unit"] is None
        assert session.evaluate("V", unit="V")["unit"] == "V"
        session.save_copy(tmp_path / "solved.mph")
    assert client.removed is True
    assert model_calls["study"] == "discharge"
    session.close()
