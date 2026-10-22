import pytest
from pydantic import ValidationError

from battery_lab.experiment import save_experiment_metadata
from battery_lab.parameter_registry import get_parameter_set_info
from battery_lab.schema import ExperimentMetadata, SimulationManifest


def test_manifest_round_trip(tmp_path):
    manifest = SimulationManifest(
        simulation_id="sim-1",
        engine="other",
        model_id="demo",
        chemistry="test",
        parameter_set_id="none",
        protocol_id="demo",
    )
    path = tmp_path / "manifest.json"
    manifest.write_json(path)
    assert path.exists()
    assert "sim-1" in path.read_text(encoding="utf-8")


def test_experiment_temperature_validation():
    meta = ExperimentMetadata(
        experiment_id="e1",
        cell_id="c1",
        chemistry="LFP",
        cell_format="pouch",
        protocol="capacity",
        temperature_K=298.15,
    )
    assert meta.temperature_K == 298.15


def test_experiment_metadata_round_trip_and_registry(tmp_path):
    meta = ExperimentMetadata(
        experiment_id="e1",
        cell_id="c1",
        chemistry="LFP",
        cell_format="pouch",
        protocol="capacity",
        temperature_K=298.15,
    )
    path = save_experiment_metadata(meta, tmp_path / "experiment.json")
    assert "e1" in path.read_text(encoding="utf-8")
    info = get_parameter_set_info("Chen2020")
    assert info.chemistry == "graphite/NMC"
    with pytest.raises(KeyError):
        get_parameter_set_info("not-a-set")
    with pytest.raises(ValidationError):
        ExperimentMetadata(
            experiment_id="e1",
            cell_id="c1",
            chemistry="LFP",
            cell_format="pouch",
            protocol="capacity",
            temperature_K=10,
        )
