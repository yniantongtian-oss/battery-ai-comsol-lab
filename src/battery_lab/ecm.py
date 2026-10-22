from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TheveninState:
    soc: float
    v_rc: float = 0.0


@dataclass(frozen=True)
class TheveninParameters:
    capacity_Ah: float
    r0_ohm: float
    r1_ohm: float
    c1_F: float


def coulomb_count(soc: float, current_A: float, dt_s: float, capacity_Ah: float) -> float:
    if not all(np.isfinite(value) for value in (soc, current_A, dt_s, capacity_Ah)):
        raise ValueError("SOC, current, time step and capacity must be finite")
    if dt_s < 0:
        raise ValueError("dt_s must be non-negative")
    if capacity_Ah <= 0:
        raise ValueError("capacity_Ah must be positive")
    updated = soc - current_A * dt_s / (capacity_Ah * 3600.0)
    return float(np.clip(updated, 0.0, 1.0))


def thevenin_step(
    state: TheveninState,
    current_A: float,
    dt_s: float,
    params: TheveninParameters,
    ocv_V: float,
) -> tuple[TheveninState, float]:
    values = (
        state.soc,
        state.v_rc,
        current_A,
        dt_s,
        params.capacity_Ah,
        params.r0_ohm,
        params.r1_ohm,
        params.c1_F,
        ocv_V,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Thevenin inputs must be finite")
    if dt_s < 0:
        raise ValueError("dt_s must be non-negative")
    if params.r0_ohm < 0 or params.r1_ohm <= 0 or params.c1_F <= 0:
        raise ValueError("r0_ohm must be non-negative; r1_ohm and c1_F must be positive")
    tau = params.r1_ohm * params.c1_F
    decay = float(np.exp(-dt_s / tau))
    v_rc = decay * state.v_rc + params.r1_ohm * (1.0 - decay) * current_A
    soc = coulomb_count(state.soc, current_A, dt_s, params.capacity_Ah)
    terminal_V = ocv_V - current_A * params.r0_ohm - v_rc
    return TheveninState(soc=soc, v_rc=v_rc), float(terminal_V)
