# Contributing

## Scientific contributions

A contribution that changes model behavior should include:

- scientific rationale,
- source/provenance for parameters,
- unit consistency,
- a validation case or regression test,
- an explicit statement of changed assumptions.

Do not commit raw proprietary datasets, credentials, licensed COMSOL binaries, or unpublished confidential parameters.

## Code workflow

Install the development + simulation stack first:

```bash
python -m pip install -e ".[dev,pybamm]"
```

Then:

1. Create a feature branch.
2. Add or update tests.
3. Run `ruff check src tests scripts`.
4. Run `python scripts/check_repository.py`.
5. Run `battery-lab validate-configs configs`.
6. Run `pytest`.
7. Keep configurations declarative; do not bury research parameters in scripts.
8. Keep notebooks exploratory. Production analysis belongs in `src/` or `scripts/`.
