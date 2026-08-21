from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    time_s = np.linspace(0, 3600, 361)
    current_A = np.full_like(time_s, 2.0)
    voltage_V = 4.15 - 1.05 * (time_s / time_s[-1]) - 0.03 * np.sin(time_s / 280)
    temperature_K = 298.15 + 2.2 * (1 - np.exp(-time_s / 900))
    pd.DataFrame(
        {
            "time_s": time_s,
            "current_A": current_A,
            "voltage_V": voltage_V,
            "temperature_K": temperature_K,
        }
    ).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
