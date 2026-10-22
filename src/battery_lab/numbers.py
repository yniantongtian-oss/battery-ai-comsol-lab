"""Strict numeric parsers for configuration values."""

from __future__ import annotations

import math
from typing import Any


def finite_float(value: Any) -> float:
    """Return a finite float, rejecting booleans and non-numeric values."""
    if isinstance(value, bool):
        raise ValueError("expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected a finite number") from exc
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def strict_int(value: Any) -> int:
    """Return a real integer, rejecting booleans and fractional floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer")
    return value
