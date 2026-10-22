from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from .config import detect_config_kind, validate_config_file
from .data_io import read_timeseries, validate_timeseries
from .schema import load_yaml

_PACKAGE_CHECKS = (
    ("numpy", True),
    ("pandas", True),
    ("pydantic", True),
    ("PyYAML", True),
    ("pybamm", False),
    ("MPh", False),
    ("scikit-learn", False),
)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _missing_dependency(name: str, install_hint: str) -> int:
    print(f"ERROR: optional runtime dependency '{name}' is not installed.", file=sys.stderr)
    print(f"Install it with: {install_hint}", file=sys.stderr)
    return 2


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def doctor_report() -> dict[str, Any]:
    packages = []
    missing_required = False
    for package, required in _PACKAGE_CHECKS:
        version = _version(package)
        packages.append(
            {
                "name": package,
                "version": version,
                "required": required,
                "installed": version is not None,
            }
        )
        missing_required = missing_required or (required and version is None)

    config_errors = []
    folders_seen = False
    for folder in ("configs", "specs"):
        root = Path(folder)
        if not root.exists():
            continue
        folders_seen = True
        for path in sorted(root.glob("*.yaml")):
            errors = validate_config_file(path)
            if errors:
                config_errors.append({"path": str(path), "errors": errors})
    return {
        "ok": not missing_required and not config_errors,
        "version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": f"{platform.system()} {platform.release()}",
        "packages": packages,
        "config_errors": config_errors,
        "configs_found": folders_seen,
    }


def _cmd_simulate(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("pybamm") is None:
        if getattr(args, "json", False):
            _print_json(
                {
                    "ok": False,
                    "errors": ["optional runtime dependency 'pybamm' is not installed"],
                    "install": 'python -m pip install -e ".[pybamm]"',
                }
            )
            return 2
        return _missing_dependency("pybamm", 'python -m pip install -e ".[pybamm]"')
    from .pybamm_runner import run

    try:
        csv_path, manifest_path = run(args.config, args.out)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if args.json:
            _print_json({"ok": False, "errors": [str(exc)]})
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(
            {
                "ok": True,
                "artifacts": {"timeseries_csv": str(csv_path), "manifest_json": str(manifest_path)},
            }
        )
    else:
        print(f"timeseries: {csv_path}")
        print(f"manifest:   {manifest_path}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("pybamm") is None:
        if getattr(args, "json", False):
            _print_json(
                {
                    "ok": False,
                    "errors": ["optional runtime dependency 'pybamm' is not installed"],
                    "install": 'python -m pip install -e ".[pybamm]"',
                }
            )
            return 2
        return _missing_dependency("pybamm", 'python -m pip install -e ".[pybamm]"')
    from .sweep import run_sweep

    try:
        summary = run_sweep(args.config)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if args.json:
            _print_json({"ok": False, "errors": [str(exc)]})
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json({"ok": True, "artifacts": {"summary_csv": str(summary)}})
    else:
        print(f"summary: {summary}")
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
    report = doctor_report()
    if getattr(args, "json", False):
        _print_json(report)
        return 0 if report["ok"] else 1
    print(f"battery-ai-comsol-lab {report['version']}")
    print(f"Python: {report['python']} ({report['executable']})")
    print(f"Platform: {report['platform']}")
    for package in report["packages"]:
        label = "required" if package["required"] else "optional"
        if package["installed"]:
            print(f"OK   {package['name']:<14} {package['version']} ({label})")
        else:
            print(f"MISS {package['name']:<14} ({label})")
    if report["config_errors"]:
        for item in report["config_errors"]:
            print(f"FAIL config {item['path']}: {'; '.join(item['errors'])}")
    elif report["configs_found"]:
        print("OK   repository configs")
    else:
        print("INFO configs/ not found from current working directory")
    pybamm_installed = any(
        package["name"] == "pybamm" and package["installed"] for package in report["packages"]
    )
    if not pybamm_installed:
        print('INFO To run simulations: python -m pip install -e ".[pybamm]"')
    print('INFO For the full research stack: python -m pip install -e ".[all]"')
    print("INFO Agent spec runner: battery-lab run-spec <spec.yaml> --json")
    print("INFO MCP server: battery-lab mcp")
    return 0 if report["ok"] else 1


def _cmd_run_spec(args: argparse.Namespace) -> int:
    from .spec import run_spec_file

    result = run_spec_file(args.spec, output_dir=args.out, dry_run=args.dry_run)
    if args.json or args.dry_run:
        _print_json(result)
    elif result["ok"]:
        print(f"ok: {result['kind']} {result['name']}")
        for key, value in result["artifacts"].items():
            print(f"{key}: {value}")
        for key, value in result["metrics"].items():
            print(f"{key}: {value}")
    else:
        for error in result["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
    if result["ok"]:
        return 0
    if any("not installed" in error for error in result["errors"]):
        return 2
    return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .spec import load_spec

    try:
        spec = load_spec(args.spec)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if spec.get("kind") != "surrogate-benchmark":
        message = "benchmark requires a spec with kind surrogate-benchmark"
        if args.json:
            _print_json({"ok": False, "errors": [message]})
        else:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return _cmd_run_spec(args)


def _cmd_mcp(args: argparse.Namespace) -> int:
    del args
    from .mcp import serve

    serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="battery-lab")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="diagnose the local installation")
    doctor.add_argument("--json", action="store_true", help="print a JSON report")
    doctor.set_defaults(func=_cmd_doctor)

    simulate = sub.add_parser("simulate", help="run a PyBaMM simulation")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--out", required=True)
    simulate.add_argument("--json", action="store_true", help="print a JSON result envelope")
    simulate.set_defaults(func=_cmd_simulate)

    sweep = sub.add_parser("sweep", help="run C-rate/temperature sweep")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--json", action="store_true", help="print a JSON result envelope")
    sweep.set_defaults(func=_cmd_sweep)

    vc = sub.add_parser("validate-configs", help="validate repository YAML configs and specs")
    vc.add_argument("path")
    vc.set_defaults(func=_cmd_validate_configs)

    vd = sub.add_parser("validate-data", help="validate canonical timeseries CSV")
    vd.add_argument("path")
    vd.set_defaults(func=_cmd_validate_data)

    run_spec = sub.add_parser("run-spec", help="run a simulation or benchmark from a YAML spec")
    run_spec.add_argument("--spec", required=True)
    run_spec.add_argument("--out", help="override the spec output_dir")
    run_spec.add_argument(
        "--dry-run", action="store_true", help="validate and resolve without executing"
    )
    run_spec.add_argument("--json", action="store_true", help="print a JSON result envelope")
    run_spec.set_defaults(func=_cmd_run_spec)

    benchmark = sub.add_parser("benchmark", help="run the physics-informed surrogate benchmark")
    benchmark.add_argument("--spec", required=True)
    benchmark.add_argument("--out", help="override the spec output_dir")
    benchmark.add_argument("--dry-run", action="store_true")
    benchmark.add_argument("--json", action="store_true")
    benchmark.set_defaults(func=_cmd_benchmark)

    mcp = sub.add_parser("mcp", help="serve the spec runner over stdio MCP")
    mcp.set_defaults(func=_cmd_mcp)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
