from __future__ import annotations

from pathlib import Path

from .schema import ExperimentMetadata


def save_experiment_metadata(metadata: ExperimentMetadata, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return path
