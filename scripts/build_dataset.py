"""Build the CityLearn-UA dataset from the installed CityLearn package.

Copies the ``citylearn_challenge_2022_phase_all`` dataset (building load/PV
profiles, weather, carbon intensity, battery parameters) and replaces the
electricity pricing with the Ukrainian residential tariff. Three pricing
variants are produced:

* ``pricing.csv``           - three-zone tariff (default; night x0.4, peaks x1.5)
* ``pricing_two_zone.csv``  - two-zone tariff (night x0.5)
* ``pricing_flat.csv``      - flat base tariff (control condition)

Zone definitions (residential meters, base price 4.32 UAH/kWh, in force
until 2026-10-31):

* three-zone: night 23:00-07:00 -> 1.728; peaks 08:00-11:00 and
  20:00-22:00 -> 6.48; other hours -> 4.32
* two-zone:   night 23:00-07:00 -> 2.16; other hours -> 4.32

Because the tariff is deterministic, the 6/12/24-hour "predicted" price
columns are exact cyclic shifts of the main series.

Usage::

    python scripts/build_dataset.py [output_dir]   # default: data/citylearn_ua

Requires: citylearn >= 2.1, pandas, numpy.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = 4.32


def three_zone(hour: int) -> float:
    if 8 <= hour < 11 or 20 <= hour < 22:
        return round(BASE * 1.5, 2)      # 6.48
    if hour >= 23 or hour < 7:
        return round(BASE * 0.4, 3)      # 1.728
    return BASE


def two_zone(hour: int) -> float:
    if hour >= 23 or hour < 7:
        return round(BASE * 0.5, 2)      # 2.16
    return BASE


def flat(hour: int) -> float:
    return BASE


def pricing_frame(price_fn, n: int) -> pd.DataFrame:
    # CityLearn 2022 time index: row t corresponds to clock hour t % 24
    prices = np.array([price_fn(t % 24) for t in range(n)])
    return pd.DataFrame({
        "electricity_pricing": prices,
        "electricity_pricing_predicted_6h": np.roll(prices, -6),
        "electricity_pricing_predicted_12h": np.roll(prices, -12),
        "electricity_pricing_predicted_24h": np.roll(prices, -24),
    })


def build(output_dir: str | Path = "data/citylearn_ua") -> Path:
    from citylearn.data import DataSet

    src = Path(DataSet.get_schema("citylearn_challenge_2022_phase_all")["root_directory"])
    dst = Path(output_dir)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    n = len(pd.read_csv(dst / "pricing.csv"))
    pricing_frame(three_zone, n).to_csv(dst / "pricing.csv", index=False)
    pricing_frame(two_zone, n).to_csv(dst / "pricing_two_zone.csv", index=False)
    pricing_frame(flat, n).to_csv(dst / "pricing_flat.csv", index=False)
    return dst


if __name__ == "__main__":
    out = build(sys.argv[1] if len(sys.argv) > 1 else "data/citylearn_ua")
    print(f"CityLearn-UA written to {out}")
