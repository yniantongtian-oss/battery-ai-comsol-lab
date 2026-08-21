from __future__ import annotations

from dataclasses import dataclass

from .metrics import mae, max_abs_error, rmse


@dataclass(frozen=True)
class ValidationSummary:
    rmse: float
    mae: float
    max_abs_error: float


def compare_series(reference, prediction) -> ValidationSummary:
    return ValidationSummary(
        rmse=rmse(reference, prediction),
        mae=mae(reference, prediction),
        max_abs_error=max_abs_error(reference, prediction),
    )
