from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_lab.comsol_bridge import ComsolSession


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase-C LGM50 COMSOL model reproducibly.")
    parser.add_argument("--model", required=True, help="Path to the local .mph model")
    parser.add_argument("--study", default="discharge_1C_298K")
    parser.add_argument("--out-model", required=True, help="Solved model copy; source is never overwritten")
    parser.add_argument("--ambient-K", type=float, default=298.15)
    parser.add_argument("--current-A", type=float, required=True)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--report", required=True, help="JSON execution manifest")
    args = parser.parse_args()

    source = Path(args.model)
    if not source.is_file():
        raise SystemExit(f"COMSOL model not found: {source}")

    out_model = Path(args.out_model)
    if out_model.resolve() == source.resolve():
        raise SystemExit("--out-model must not overwrite the source .mph")

    out_model.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with ComsolSession(cores=args.cores) as session:
        session.open_model(source)
        session.set_parameters({"T_amb": args.ambient_K, "I_app": args.current_A})
        session.run_study(args.study)
        session.save_copy(out_model)

    report = {
        "source_model": str(source),
        "solved_copy": str(out_model),
        "study": args.study,
        "parameters": {"T_amb": args.ambient_K, "I_app": args.current_A},
        "status": "solved",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
