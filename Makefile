.PHONY: install install-all doctor test lint check demo simulate

install:
	python -m pip install -e ".[dev,pybamm]"

install-all:
	python -m pip install -e ".[all]"

doctor:
	battery-lab doctor

test:
	pytest

lint:
	ruff check src tests scripts

check: lint test
	python scripts/check_repository.py
	battery-lab validate-configs configs

demo:
	python scripts/generate_demo_data.py --out data/processed/demo_cycle.csv

simulate:
	battery-lab simulate --config configs/lgm50_electrothermal_baseline.yaml --out runs/lgm50_baseline
