import pytest

pybamm = pytest.importorskip("pybamm")

from battery_lab.pybamm_runner import build_model


def test_oregan_dfn_lumped_processes():
    model = build_model({"model": "DFN", "thermal": "lumped"})
    params = pybamm.ParameterValues("ORegan2022")
    params.process_model(model)
