# LGM50 reference 3D electro-thermal model card — v0

## Status

**Model card and automation contract only.** The binary `.mph` model is intentionally not committed and this repository does not claim that a COMSOL solve has been executed in CI. A local COMSOL Multiphysics installation and license are required.

## Scientific question

Quantify the effect of cylindrical geometry, current-collector/tab representation, and external thermal boundary conditions on terminal voltage, heat generation, and spatial temperature gradients for the repository's LGM50/ORegan2022 reference track.

## Traceability

- Target-cell definition: `configs/target_cell_lgm50_reference.yaml`
- Cell-level reference: `configs/lgm50_electrothermal_baseline.yaml`
- Parameter family: `ORegan2022`
- Cross-validation utility: `scripts/cross_validate_pybamm_comsol.py`
- Local binary target: `comsol/local_models/lgm50_3d_electrothermal_v0.mph`

## Required named parameters

The local model should expose stable parameter names rather than entity-number assumptions:

- `T_amb` — ambient temperature
- `I_app` — applied current
- `h_side`, `h_top`, `h_bottom` — convective coefficients
- `k_r`, `k_z` — effective radial/axial thermal conductivity
- `rho_eff`, `cp_eff` — effective density and heat capacity

Every calibrated value must record units, source, and revision in the model notes or an adjacent parameter manifest.

## Required named selections

At minimum:

- `cell_domain`
- `jellyroll_domain`
- `positive_tab`
- `negative_tab`
- `can_surface`
- `top_surface`
- `bottom_surface`

Use named selections in physics, mesh, probes, and post-processing. Do not bind automation to fragile entity IDs.

## Physics and coupling

The first validated revision should include:

1. an electrochemical or reduced electrochemical heat-source representation traceable to the cell baseline,
2. anisotropic effective thermal transport where justified,
3. current collector/tab electrical loss representation,
4. convective boundary-condition variants for side/top/bottom cooling,
5. terminal-voltage and heat-generation probes,
6. volume-average and maximum-temperature probes.

## Study contract

Create a named transient study `discharge_1C_298K` reproducing the baseline 1C, 298.15 K discharge condition before adding more aggressive sweeps.

The automation path must be able to:

1. load the model through `battery_lab.comsol_bridge.ComsolSession`,
2. set named parameters,
3. run the named study,
4. export a table with `time_s`, `voltage_V`, `temperature_K`,
5. save a copy of the solved model without overwriting the source asset.

## Mesh convergence gate

Use at least three mesh levels. Record element count and the selected observables for each level. The chosen production mesh must have a documented convergence criterion justified by the scientific question; do not hard-code a universal percentage.

## Cross-validation gate

Compare the overlapping time interval against the trusted PyBaMM or experimental reference with:

```bash
python scripts/cross_validate_pybamm_comsol.py \
  --reference runs/lgm50_baseline/timeseries.csv \
  --candidate comsol/exports/lgm50_3d_electrothermal_v0.csv \
  --out runs/cross_validation/lgm50_3d_report.json
```

Acceptance thresholds must come from `configs/target_cell_lgm50_reference.yaml` after experimental repeatability and instrument capability are established.

## Phase-C exit evidence

Phase C is not complete until the repository has all of the following:

- reproducible local `.mph` revision
- parameter provenance
- mesh-convergence record
- scripted run/export path
- PyBaMM-versus-COMSOL report
- solver settings and software versions
- at least one cooling boundary-condition comparison
