from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from importlib import metadata
from pathlib import Path

from . import __version__
from .config import detect_config_kind, validate_config_file
from .data_io import read_timeseries, validate_timeseries
from .schema import load_yaml


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _missing_dependency(name: str, install_hint: str) -> int:
    print(f"ERROR: optional runtime dependency '{name}' is not installed.", file=sys.stderr)
    print(f"Install it with: {install_hint}", file=sys.stderr)
    return 2


def _cmd_simulate(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("pybamm") is None:
        return _missing_dependency("pybamm", 'python -m pip install -e ".[pybamm]"')
    from .pybamm_runner import run

    csv_path, manifest_path = run(args.config, args.out)
    print(f"timeseries: {csv_path}")
    print(f"manifest:   {manifest_path}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("pybamm") is None:
        return _missing_dependency("pybamm", 'python -m pip install -e ".[pybamm]"')
    from .sweep import run_sweep

    print(f"summary: {run_sweep(args.config)}")
    return 0


def _cmd_validate_configs(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.exists():
        print(f"FAIL {root}: path does not exist")
        return 1
    paths = [root] if root.is_file() else sorted(root.glob("*.yaml"))
    if not paths:
        print(f"FAIL {root}: no YAML configs found")
        return 1

    failures = 0
    for path in paths:
        config = load_yaml(path)
        kind = detect_config_kind(config)
        errors = validate_config_file(path)
        if errors:
            failures += 1
            print(f"FAIL [{kind:10}] {path}: {'; '.join(errors)}")
        else:
            print(f"OK   [{kind:10}] {path}")
    return 1 if failures else 0


def _cmd_validate_data(args: argparse.Namespace) -> int:
    errors = validate_timeseries(read_timeseries(args.path))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("OK")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    del args
    print(f"battery-ai-comsol-lab {__version__}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.release()}")

    checks = [
        ("numpy", True),
        ("pandas", True),
        ("pydantic", True),
        ("PyYAML", True),
        ("pybamm", False),
        ("MPh", False),
        ("scikit-learn", False),
    ]
    missing_required = False
    for package, required in checks:
        version = _version(package)
        label = "required" if required else "optional"
        if version:
            print(f"OK   {package:<14} {version} ({label})")
        else:
            print(f"MISS {package:<14} ({label})")
            missing_required |= required

    config_root = Path("configs")
    if config_root.exists():
        config_failures = []
        for path in sorted(config_root.glob("*.yaml")):
            errors = validate_config_file(path)
            if errors:
                config_failures.append((path, errors))
        if config_failures:
            for path, errors in config_failures:
                print(f"FAIL config {path}: {'; '.join(errors)}")
        else:
            print("OK   repository configs")
    else:
        print("INFO configs/ not found from current working directory")

    pybamm_version = _version("pybamm")
    if pybamm_version is None:
        print('INFO To run simulations: python -m pip install -e ".[pybamm]"')
    print('INFO For the full research stack: python -m pip install -e ".[all]"')
    return 1 if missing_required else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="battery-lab")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="diagnose the local installation")
    doctor.set_defaults(func=_cmd_doctor)

    simulate = sub.add_parser("simulate", help="run a PyBaMM simulation")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--out", required=True)
    simulate.set_defaults(func=_cmd_simulate)

    sweep = sub.add_parser("sweep", help="run C-rate/temperature sweep")
    sweep.add_argument("--config", required=True)
    sweep.set_defaults(func=_cmd_sweep)

    vc = sub.add_parser("validate-configs", help="validate repository YAML configs")
    vc.add_argument("path")
    vc.set_defaults(func=_cmd_validate_configs)

    vd = sub.add_parser("validate-data", help="validate canonical timeseries CSV")
    vd.add_argument("path")
    vd.set_defaults(func=_cmd_validate_data)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
