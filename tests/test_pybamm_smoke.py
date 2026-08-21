import importlib

import pytest

pybamm = pytest.importorskip("pybamm")
build_model = importlib.import_module("battery_lab.pybamm_runner").build_model


def test_oregan_dfn_lumped_processes():
    model = build_model({"model": "DFN", "thermal": "lumped"})
    params = pybamm.ParameterValues("ORegan2022")
    params.process_model(model)
