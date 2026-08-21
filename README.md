# Battery AI + COMSOL Lab

A reproducible research repository for **battery electrochemistry, multiphysics simulation, experiments, AI/ML, BMS algorithms, pack studies, and digital twins**.

The project is designed around one rule: every result should be traceable from raw inputs to a versioned model, configuration, dataset, metric, and code commit.

## What this repository covers

- Physics-based cell models: ECM, SPM, SPMe, DFN/P2D
- Electro-thermal coupling and temperature sweeps
- COMSOL 1D/2D/3D multiphysics orchestration through MPh
- Ageing research: SEI, lithium plating, LAM, resistance growth
- Mechanical research: swelling, diffusion-induced stress, constraints
- Thermal management and pack-level temperature gradients
- Safety modeling and mitigation-oriented thermal-runaway research
- Experimental data organization: capacity, HPPC/pulse, EIS, GITT/PITT, ageing
- Parameter identification, sensitivity and identifiability
- AI surrogate models, SOH/RUL, active learning and uncertainty workflows
- BMS state estimation and digital-twin state synchronization
- Multi-objective design and Pareto screening

## Repository layout

```text
.
├── configs/                 # Reproducible model and sweep configurations
├── data/                    # Data contracts and local data folders
├── docs/                    # Research program and modeling specifications
├── experiments/             # Experiment metadata/templates
├── notebooks/               # Exploratory work only
├── papers/                  # Paper-reproduction registry
├── scripts/                 # Validation/demo/publish helpers
├── src/battery_lab/         # Python research package
├── tests/                   # Unit + smoke tests
└── .github/workflows/       # CI
```

## Model ladder

Use the cheapest model that can answer the scientific question.

| Level | Model | Use |
|---|---|---|
| L0 | Rint / Thevenin ECM | control, online estimation, pack studies |
| L1 | SPM | fast physics screening |
| L2 | SPMe | electrolyte effects at moderate cost |
| L3 | DFN/P2D | detailed cell electrochemistry |
| L4 | DFN + thermal | electro-thermal validation |
| L5 | COMSOL 2D/3D | tabs, local gradients, cooling, mechanics |
| L6 | ROM / surrogate | optimization and real-time digital twins |

## Baselines included

1. **LFP/graphite electrochemical baseline** using PyBaMM `Prada2013` with isothermal DFN.
2. **High-energy cylindrical electro-thermal baseline** using `ORegan2022` with lumped thermal DFN.
3. **NMC/graphite baseline** using `Chen2020`.
4. **Ageing-oriented baseline** using `OKane2022` degradation-capable options.

The parameter sets are research references, not specifications for any commercial cell.

## Install

PyBaMM currently supports Python 3.10-3.14. Python 3.11, 3.12, or 3.13 is a good default for this repository.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[pybamm]"
battery-lab doctor
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[pybamm]"
battery-lab doctor
```

For development tools, use `python -m pip install -e ".[dev,pybamm]"`. For COMSOL + ML + development dependencies, use `python -m pip install -e ".[all]"`. A valid local COMSOL installation and license are still required for COMSOL execution.

The configuration and data-validation commands do not require PyBaMM, so `python -m pip install -e .` is enough if you only want the lightweight tooling.

## Quick start

Check the installation first:

```bash
battery-lab doctor
```

Validate every repository configuration. The command automatically distinguishes simulation, sweep, and pack configs:

```bash
battery-lab validate-configs configs
```

Run a PyBaMM baseline:

```bash
battery-lab simulate --config configs/lgm50_electrothermal_baseline.yaml --out runs/lgm50_baseline
```

Run a parameter sweep:

```bash
battery-lab sweep --config configs/lgm50_crate_temperature_sweep.yaml
```

Create safe synthetic/demo data and validate it:

```bash
python scripts/generate_demo_data.py --out data/processed/demo_cycle.csv
battery-lab validate-data data/processed/demo_cycle.csv
```

You can also run the CLI without the installed console script:

```bash
python -m battery_lab --help
```

### Run directly on GitHub

Open **Actions -> Run battery simulation -> Run workflow**, choose a baseline, and GitHub will install PyBaMM, run the model, and upload `timeseries.csv` plus `manifest.json` as a downloadable workflow artifact. No local COMSOL installation is needed for PyBaMM runs.

## Research output contract

Every publishable run should record:

- chemistry and cell format
- parameter-set name and source
- model family and options
- geometry revision when applicable
- initial/boundary conditions
- solver settings
- experiment/protocol ID
- raw-data provenance
- preprocessing version
- software versions
- code commit SHA
- metrics and uncertainty information

The `SimulationManifest` written by the runner implements this contract.

## COMSOL strategy

COMSOL files are intentionally treated as **versioned scientific assets**, not opaque click-built files. Each `.mph` model should have:

- a human-readable model card
- parameter table with units and provenance
- named physics interfaces and studies
- named selections rather than fragile entity IDs
- mesh-convergence evidence
- solver settings recorded in the model manifest
- exported CSV/HDF5 result tables for cross-validation

See `docs/02-comsol-modeling.md`.

## Scientific maturity gates

A research track is mature only when it has:

1. a precise question,
2. a reproducible baseline,
3. sensitivity/identifiability analysis,
4. experiment or trusted-reference validation,
5. uncertainty/error reporting,
6. at least one competing-model or ablation comparison,
7. a scripted path from inputs to final figures/tables.

## Safety boundary

The safety track focuses on simulation, detection, prevention, containment, and mitigation. The repository does not provide procedural instructions for intentionally inducing hazardous battery failures.

## Status

`v0.1.0` — complete research architecture + runnable PyBaMM pipeline + COMSOL adapter + data contracts + BMS/digital-twin primitives + CI.
