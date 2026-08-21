# COMSOL Modeling Specification

## Model hierarchy

1. 1D porous-electrode reference model.
2. 2D current-collector/tab geometry.
3. 3D cell thermal model.
4. Electrochemical + thermal coupling.
5. Mechanics/swelling extension.
6. Module/pack heat-transfer model.

## Required model metadata

Every COMSOL model should have a sidecar model card containing:

- model ID and revision
- COMSOL version/modules
- chemistry and cell format
- geometry source
- parameter table and units
- selections and coordinate conventions
- physics interfaces
- boundary and initial conditions
- mesh strategy and mesh-convergence results
- study/solver configuration
- exported observables
- validation references

## Naming conventions

Use stable named selections and parameter names. Avoid scripts that depend on automatically assigned domain/boundary numbers.

Recommended parameter prefixes:

- `geo_` geometry
- `ec_` electrochemistry
- `th_` thermal
- `mech_` mechanics
- `exp_` experiment-derived
- `cal_` calibrated

## Cross-validation

For matched protocols, compare PyBaMM and COMSOL on:

- terminal voltage
- average/max temperature
- heat generation
- selected concentration and potential outputs

Use explicit RMSE/MAE/max-error metrics and document differences in governing assumptions.

## Automation

The Python layer deliberately exposes a small deterministic API: open, set parameters, solve, evaluate, save. Agentic behavior should operate above this layer rather than manipulating the GUI directly.
