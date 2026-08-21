.PHONY: install test lint check demo

install:
	python -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check src tests scripts

check: lint test

demo:
	python scripts/generate_demo_data.py --out data/processed/demo_cycle.csv
