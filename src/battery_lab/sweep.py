from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .pybamm_runner import run
from .schema import load_yaml


def run_sweep(sweep_config_path: str | Path) -> Path:
    sweep_config_path = Path(sweep_config_path)
    sweep = load_yaml(sweep_config_path)

    base_path = Path(str(sweep["base_config"]))
    if not base_path.is_absolute():
        base_path = sweep_config_path.parent / base_path
    base = load_yaml(base_path)

    c_rates = [float(value) for value in sweep.get("c_rates", [1.0])]
    temperatures = [float(v) for v in sweep.get("ambient_temperatures_K", [298.15])]
    output_root = Path(str(sweep.get("output_root", "runs/sweep")))
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for c_rate in c_rates:
        for temperature_K in temperatures:
            run_name = f"C{c_rate:g}_T{temperature_K:.2f}K".replace(".", "p")
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            config = dict(base)
            config["c_rate"] = c_rate
            config["ambient_temperature_K"] = temperature_K
            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            csv_path, manifest_path = run(config_path, run_dir)
            frame = pd.read_csv(csv_path)
            rows.append(
                {
                    "run": run_name,
                    "c_rate": c_rate,
                    "ambient_temperature_K": temperature_K,
                    "end_time_s": float(frame["time_s"].iloc[-1]),
                    "end_voltage_V": float(frame["voltage_V"].iloc[-1]),
                    "max_temperature_K": float(frame["temperature_K"].max()),
                    "timeseries_csv": str(csv_path),
                    "manifest_json": str(manifest_path),
                }
            )

    summary_path = output_root / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path
