from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from battery_lab.cross_validation import compare_result_tables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare COMSOL exports against a trusted reference time series."
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference CSV, typically PyBaMM or experiment",
    )
    parser.add_argument("--candidate", required=True, help="Candidate CSV, typically COMSOL export")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["voltage_V", "temperature_K"],
        help="Signals to compare after time interpolation",
    )
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    candidate = pd.read_csv(args.candidate)
    report = compare_result_tables(reference, candidate, signals=args.signals)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
