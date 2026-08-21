from __future__ import annotations

import argparse
from pathlib import Path

from .config import validate_config_file
from .data_io import read_timeseries, validate_timeseries
from .pybamm_runner import run
from .sweep import run_sweep


def _cmd_simulate(args: argparse.Namespace) -> int:
    csv_path, manifest_path = run(args.config, args.out)
    print(f"timeseries: {csv_path}")
    print(f"manifest:   {manifest_path}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    print(f"summary: {run_sweep(args.config)}")
    return 0


def _cmd_validate_configs(args: argparse.Namespace) -> int:
    root = Path(args.path)
    paths = [root] if root.is_file() else sorted(root.glob("*.yaml"))
    failures = 0
    for path in paths:
        errors = validate_config_file(path)
        if errors:
            failures += 1
            print(f"FAIL {path}: {'; '.join(errors)}")
        else:
            print(f"OK   {path}")
    return 1 if failures else 0


def _cmd_validate_data(args: argparse.Namespace) -> int:
    errors = validate_timeseries(read_timeseries(args.path))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="battery-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="run a PyBaMM simulation")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--out", required=True)
    simulate.set_defaults(func=_cmd_simulate)

    sweep = sub.add_parser("sweep", help="run C-rate/temperature sweep")
    sweep.add_argument("--config", required=True)
    sweep.set_defaults(func=_cmd_sweep)

    vc = sub.add_parser("validate-configs", help="validate YAML configs")
    vc.add_argument("path")
    vc.set_defaults(func=_cmd_validate_configs)

    vd = sub.add_parser("validate-data", help="validate canonical timeseries CSV")
    vd.add_argument("path")
    vd.set_defaults(func=_cmd_validate_data)

    args = parser.parse_args()
    raise SystemExit(args.func(args))
