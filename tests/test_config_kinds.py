from pathlib import Path

from battery_lab.config import detect_config_kind, validate_config_file
from battery_lab.schema import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_detects_simulation_sweep_and_pack_configs():
    lfp = load_yaml(ROOT / "configs/lfp_graphite_baseline.yaml")
    assert detect_config_kind(lfp) == "simulation"
    sweep = load_yaml(ROOT / "configs/lgm50_crate_temperature_sweep.yaml")
    assert detect_config_kind(sweep) == "sweep"
    assert detect_config_kind(load_yaml(ROOT / "configs/pack_research_template.yaml")) == "pack"


def test_every_repository_config_is_valid():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        assert validate_config_file(path) == [], path
