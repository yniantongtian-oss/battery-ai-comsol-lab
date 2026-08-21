# AI / ML Program

## Research targets

- surrogate prediction of voltage/temperature/field summaries
- SOH and RUL estimation
- parameter inversion from experiment data
- active-learning selection of expensive COMSOL runs
- uncertainty-aware optimization
- reduced-order models for digital twins

## Dataset discipline

Split by cell, experiment or ageing trajectory—not by randomly shuffling time samples from the same trajectory. Prevent leakage from future cycles, replicate cells or preprocessing statistics.

## Baselines

Always include simple baselines before deep models:

- linear/ridge regression
- equivalent-circuit estimation
- tree models where appropriate
- physics-based residual models

## Evaluation

Report absolute metrics, normalized metrics, calibration/uncertainty where relevant, extrapolation tests and failure cases. A visually smooth curve is not evidence of generalization.

## Physics-informed methods

PINNs and neural operators are research tracks, not default replacements for trusted solvers. Demonstrate a clear advantage in speed, inverse problems, sparse observations or differentiability.
