from battery_lab.digital_twin import BatteryDigitalTwin, TwinObservation
from battery_lab.ecm import TheveninParameters


def test_digital_twin_updates_soc_temperature_and_residual():
    twin = BatteryDigitalTwin(
        TheveninParameters(capacity_Ah=2.0, r0_ohm=0.01, r1_ohm=0.02, c1_F=1000),
        initial_soc=0.8,
    )
    observation = TwinObservation(
        current_A=1.0, voltage_V=3.5, temperature_K=300.0, dt_s=1.0
    )
    state = twin.update(observation, 3.7)
    assert state.ecm.soc < 0.8
    assert state.temperature_K == 300.0
    assert state.residual_V != 0.0
    assert twin.state is state
