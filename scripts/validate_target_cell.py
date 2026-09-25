from __future__ import annotations

import argparse
from pathlib import Path

from battery_lab.target_cell import evidence_readiness, load_target_cell_spec, validate_target_cell_spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Phase-B target-cell evidence definition.")
    parser.add_argument("spec")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    spec = load_target_cell_spec(args.spec)
    errors = validate_target_cell_spec(spec, root=Path(args.root))
    readiness = evidence_readiness(spec)
    for name, status in readiness.items():
        print(f"{name}: {status}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
