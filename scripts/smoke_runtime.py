from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from battery_lab.pybamm_runner import run
from battery_lab.schema import load_yaml
from battery_lab.sweep import run_sweep

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (
    "lfp_graphite_baseline.yaml",
    "nmc_graphite_baseline.yaml",
    "lgm50_electrothermal_baseline.yaml",
    "ageing_okane2022.yaml",
)


def _write_short_config(source: Path, target: Path) -> None:
    config = load_yaml(source)
    config["max_duration_s"] = 60
    config["output_points"] = 12
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="battery-lab-smoke-") as tmp:
        work = Path(tmp)
        for filename in BASELINES:
            config_path = work / filename
            _write_short_config(ROOT / "configs" / filename, config_path)
            out_dir = work / filename.removesuffix(".yaml")
            csv_path, manifest_path = run(config_path, out_dir)
            frame = pd.read_csv(csv_path)
            if frame.empty or not manifest_path.exists():
                raise RuntimeError(f"runtime smoke failed for {filename}")
            print(f"OK baseline {filename}: {len(frame)} samples")

        sweep_base = work / "sweep_base.yaml"
        _write_short_config(ROOT / "configs" / "lgm50_electrothermal_baseline.yaml", sweep_base)
        sweep_config = work / "sweep.yaml"
        sweep_config.write_text(
            yaml.safe_dump(
                {
                    "base_config": sweep_base.name,
                    "c_rates": [0.5, 1.0],
                    "ambient_temperatures_K": [298.15],
                    "output_root": str(work / "sweep-runs"),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        summary_path = run_sweep(sweep_config)
        summary = pd.read_csv(summary_path)
        if len(summary) != 2:
            raise RuntimeError(f"sweep smoke expected 2 runs, got {len(summary)}")
        print(f"OK sweep: {len(summary)} runs")


if __name__ == "__main__":
    main()
