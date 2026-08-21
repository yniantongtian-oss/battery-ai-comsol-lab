from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSetInfo:
    name: str
    chemistry: str
    preferred_thermal: str
    purpose: str


PARAMETER_SETS = {
    "Prada2013": ParameterSetInfo(
        "Prada2013", "LFP/graphite", "isothermal", "LFP electrochemical reference"
    ),
    "ORegan2022": ParameterSetInfo(
        "ORegan2022", "graphite/NMC811", "lumped", "electro-thermal cylindrical reference"
    ),
    "Chen2020": ParameterSetInfo(
        "Chen2020", "graphite/NMC", "isothermal", "high-energy electrochemical reference"
    ),
    "OKane2022": ParameterSetInfo(
        "OKane2022", "graphite/NMC", "lumped", "degradation research reference"
    ),
}


def get_parameter_set_info(name: str) -> ParameterSetInfo:
    try:
        return PARAMETER_SETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown curated parameter set {name!r}") from exc
