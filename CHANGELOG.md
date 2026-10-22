# Changelog

## 0.2.0 - 2026-10-22

- Require PyBaMM 26.10.1, NumPy 2, and current scientific-Python floors. COMSOL stays optional through MPh 1.4.
- Reject non-finite and non-integer configuration values instead of truncating or accepting NaN.
- Resample PyBaMM solutions onto the requested output grid, record early voltage-cutoff stops, and keep isothermal temperature on the data contract.
- Build `x-lumped` models with pouch geometry, which PyBaMM 26 requires.
- Add a seeded, protocol-split physics-informed surrogate benchmark with hashed datasets and thresholded metrics.
- Add `battery-lab run-spec`, `battery-lab benchmark`, and a stdio MCP server for agents.

## 0.1.0 - 2026-08-21

- Independent battery research repository architecture.
- PyBaMM SPM/SPMe/DFN runner and sweep engine.
- LFP, high-energy electro-thermal, NMC and ageing configurations.
- Reproducible manifest/data contracts.
- COMSOL/MPh deterministic bridge.
- Metrics, parameter registry, experiment metadata and validation helpers.
- ECM/BMS and digital-twin primitives.
- Lightweight surrogate and Pareto utilities.
- CI, tests, research documentation and publication helpers.
