from __future__ import annotations

from pathlib import Path
from typing import Any


class ComsolUnavailableError(RuntimeError):
    """Raised when COMSOL/MPh is unavailable in the current environment."""


class ComsolSession:
    """Narrow deterministic adapter around MPh for reproducible COMSOL workflows."""

    def __init__(self, cores: int | None = None) -> None:
        try:
            import mph
        except ImportError as exc:
            raise ComsolUnavailableError(
                'Install the optional dependency with `python -m pip install -e ".[comsol]"` '
                "and ensure COMSOL is installed and licensed locally."
            ) from exc
        self._mph = mph
        kwargs: dict[str, Any] = {}
        if cores is not None:
            kwargs["cores"] = cores
        self.client = mph.start(**kwargs)
        self.model = None

    def open_model(self, path: str | Path) -> None:
        self.model = self.client.load(str(Path(path)))

    def _require_model(self):
        if self.model is None:
            raise RuntimeError("No COMSOL model is open")
        return self.model

    def set_parameters(self, values: dict[str, str | int | float]) -> None:
        model = self._require_model()
        for name, value in values.items():
            model.parameter(name, str(value))

    def run_study(self, study_name: str | None = None) -> None:
        self._require_model().solve(study_name)

    def evaluate(self, expressions: str | list[str], unit: str | None = None):
        return self._require_model().evaluate(expressions, unit=unit)

    def save_copy(self, path: str | Path) -> None:
        self._require_model().save(str(Path(path)))

    def close_model(self) -> None:
        if self.model is not None:
            try:
                self.client.remove(self.model)
            finally:
                self.model = None

    def __enter__(self) -> ComsolSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_model()
