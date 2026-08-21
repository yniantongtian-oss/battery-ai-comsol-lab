from __future__ import annotations

from dataclasses import dataclass

from .ecm import TheveninParameters, TheveninState, thevenin_step


@dataclass
class TwinObservation:
    current_A: float
    voltage_V: float
    temperature_K: float
    dt_s: float


@dataclass
class TwinState:
    ecm: TheveninState
    temperature_K: float
    residual_V: float = 0.0


class BatteryDigitalTwin:
    def __init__(self, params: TheveninParameters, initial_soc: float = 1.0) -> None:
        self.params = params
        self.state = TwinState(ecm=TheveninState(initial_soc), temperature_K=298.15)

    def update(self, obs: TwinObservation, ocv_V: float) -> TwinState:
        ecm_state, predicted_V = thevenin_step(
            self.state.ecm, obs.current_A, obs.dt_s, self.params, ocv_V
        )
        self.state = TwinState(
            ecm=ecm_state,
            temperature_K=obs.temperature_K,
            residual_V=obs.voltage_V - predicted_V,
        )
        return self.state
