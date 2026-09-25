from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .validation import compare_series

DEFAULT_SIGNALS = ("voltage_V", "temperature_K")


def _prepare(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    if time_column not in frame:
        raise ValueError(f"missing time column: {time_column}")
    out = frame.copy()
    out[time_column] = pd.to_numeric(out[time_column], errors="coerce")
    out = out[np.isfinite(out[time_column])].sort_values(time_column)
    if out.empty:
        raise ValueError("no finite time samples")
    if out[time_column].duplicated().any():
        out = out.groupby(time_column, as_index=False).mean(numeric_only=True)
    return out


def compare_result_tables(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    signals: Iterable[str] = DEFAULT_SIGNALS,
    time_column: str = "time_s",
) -> dict[str, object]:
    """Interpolate candidate results onto the reference time grid and score them."""
    ref = _prepare(reference, time_column)
    cand = _prepare(candidate, time_column)

    start = max(float(ref[time_column].min()), float(cand[time_column].min()))
    stop = min(float(ref[time_column].max()), float(cand[time_column].max()))
    if stop <= start:
        raise ValueError("reference and candidate time ranges do not overlap")

    mask = (ref[time_column] >= start) & (ref[time_column] <= stop)
    ref_overlap = ref.loc[mask].copy()
    if len(ref_overlap) < 2:
        raise ValueError("fewer than two reference samples in the overlap window")

    result: dict[str, object] = {
        "time_overlap_s": [start, stop],
        "reference_samples": int(len(ref_overlap)),
        "candidate_samples": int(len(cand)),
        "signals": {},
    }

    t_ref = ref_overlap[time_column].to_numpy(dtype=float)
    t_cand = cand[time_column].to_numpy(dtype=float)

    signal_reports: dict[str, object] = {}
    for signal in signals:
        if signal not in ref_overlap or signal not in cand:
            signal_reports[signal] = {"status": "missing"}
            continue

        y_ref = pd.to_numeric(ref_overlap[signal], errors="coerce").to_numpy(dtype=float)
        y_cand = pd.to_numeric(cand[signal], errors="coerce").to_numpy(dtype=float)
        finite_cand = np.isfinite(t_cand) & np.isfinite(y_cand)
        finite_ref = np.isfinite(y_ref)
        if finite_cand.sum() < 2 or finite_ref.sum() < 2:
            signal_reports[signal] = {"status": "insufficient_finite_data"}
            continue

        interp = np.interp(t_ref[finite_ref], t_cand[finite_cand], y_cand[finite_cand])
        summary = compare_series(y_ref[finite_ref], interp)
        signal_reports[signal] = {
            "status": "ok",
            **asdict(summary),
            "samples": int(finite_ref.sum()),
        }

    result["signals"] = signal_reports
    return result
