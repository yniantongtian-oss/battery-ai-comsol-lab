from battery_lab.config import validate_simulation_config


def test_good_config():
    errors = validate_simulation_config(
        {
            "chemistry": "LFP_graphite",
            "engine": "pybamm",
            "model": "DFN",
            "thermal": "isothermal",
            "parameter_set": "Prada2013",
            "protocol_id": "cc",
            "c_rate": 1.0,
            "ambient_temperature_K": 298.15,
        }
    )
    assert errors == []


def test_bad_crate():
    errors = validate_simulation_config(
        {
            "chemistry": "x",
            "engine": "pybamm",
            "model": "DFN",
            "parameter_set": "x",
            "protocol_id": "x",
            "c_rate": 0,
        }
    )
    assert any("c_rate" in error for error in errors)
