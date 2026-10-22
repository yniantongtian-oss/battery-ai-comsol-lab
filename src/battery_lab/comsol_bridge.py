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
        try:
            self.client = mph.start(**kwargs)
        except (OSError, RuntimeError) as exc:
            raise ComsolUnavailableError(
                "COMSOL is not available in this environment. Install and license COMSOL "
                "locally; the rest of the lab runs without it."
            ) from exc
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
        model = self._require_model()
        if unit is None:
            return model.evaluate(expressions)
        return model.evaluate(expressions, unit=unit)

    def save_copy(self, path: str | Path) -> None:
        self._require_model().save(str(Path(path)))

    def close_model(self) -> None:
        if self.model is None:
            return
        try:
            self.client.remove(self.model)
        finally:
            self.model = None

    def close(self) -> None:
        """Remove this session's model. The shared COMSOL client is left running."""
        self.close_model()

    def __enter__(self) -> ComsolSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
