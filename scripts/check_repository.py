from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from battery_lab.config import validate_config_file  # noqa: E402


def main() -> None:
    failures = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        errors = validate_config_file(path)
        if errors:
            failures.append((path, errors))
    if failures:
        for path, errors in failures:
            print(path.relative_to(ROOT), errors)
        raise SystemExit(1)
    print("repository configuration checks passed")


if __name__ == "__main__":
    main()
