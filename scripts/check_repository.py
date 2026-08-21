from __future__ import annotations

from pathlib import Path

from battery_lab.config import validate_config_file


def main() -> None:
    failures = []
    for path in sorted(Path("configs").glob("*.yaml")):
        if path.name.endswith("sweep.yaml") or "pack_research" in path.name:
            continue
        errors = validate_config_file(path)
        if errors:
            failures.append((path, errors))
    if failures:
        for path, errors in failures:
            print(path, errors)
        raise SystemExit(1)
    print("repository configuration checks passed")


if __name__ == "__main__":
    main()
