# BMS and Digital Twin

## Layers

1. Fast ECM for control-loop prediction.
2. SOC state estimation.
3. SOH/capacity/resistance adaptation.
4. Temperature/state residual monitoring.
5. Slow physics model for recalibration and scenario analysis.

The included `TheveninState` and `BatteryDigitalTwin` are intentionally small baselines. Production BMS work should add observability analysis, filters, sensor bias models, fault handling and rigorous validation.

## Synchronization

A digital twin should distinguish measured states, estimated states, calibrated parameters and forecast states. Residuals should remain visible rather than being silently absorbed by parameter tuning.
