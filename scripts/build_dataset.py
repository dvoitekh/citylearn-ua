"""Build the CityLearn-UA dataset from the installed CityLearn package.

Starts from ``citylearn_challenge_2022_phase_all`` (building load profiles,
weather, carbon intensity, battery parameters) and applies the Ukrainian
adaptations:

* **Pricing** — Ukrainian residential tariff, three variants:
  ``pricing.csv`` (three-zone, default), ``pricing_two_zone.csv``,
  ``pricing_flat.csv``. Base price 4.32 UAH/kWh (in force until 2026-10-31);
  night 23:00-07:00 x0.4, peaks 08:00-11:00 and 20:00-22:00 x1.5.
* **PV generation** (``--pv kyiv``, default) — the ``solar_generation`` column
  of every building is replaced with a real Kyiv year from PVGIS (SARAH2,
  2019, 1 kWp, 35 deg tilt, south, 14% system loss): ~1157 kWh/kWp/year
  against ~1600 for the source Texas profile, with a 4.5x winter dip.
  ``--pv texas`` keeps the original profile (climate as a free variable).
* **Outages** (``--outages``, default ``none``) — see ``make_outage_schedules.py``.

Alignment notes
---------------
The CityLearn 2022 calendar runs August->July: row 0 is 31 July, hour 24
(i.e. 23:00-24:00), and ``hour`` is 1..24. External series are converted to
local time and rolled to that start. PVGIS timestamps are UTC; Kyiv is
UTC+2 (EET) and daylight saving is **not** modelled - a documented
simplification worth at most a one-hour shift in summer.

Usage::

    python scripts/build_dataset.py [output_dir] [--pv kyiv|texas]
                                    [--outages none|scheduled_4x2|stochastic]

Requires: citylearn >= 2.1, pandas, numpy (requests optional: the raw PVGIS
response is cached in data/raw/, so a network connection is not needed).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"

BASE = 4.32
PVGIS_URL = (
    "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    "?lat=50.45&lon=30.52&pvcalculation=1&peakpower=1&loss=14"
    "&angle=35&aspect=0&outputformat=json&startyear=2019&endyear=2019"
)
KYIV_UTC_OFFSET = 2  # EET, DST not modelled
# Dataset row 0 = 31 July, hour 24 -> index in a Jan-1-based local hourly year
DATASET_START_INDEX = (pd.Timestamp("2019-07-31").dayofyear - 1) * 24 + 23


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

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
    # Dataset row t corresponds to clock hour t % 24 (row 0 = hour 24 == 0)
    prices = np.array([price_fn(t % 24) for t in range(n)])
    return prices_to_frame(prices)


def prices_to_frame(prices: np.ndarray) -> pd.DataFrame:
    """Wrap a price series into CityLearn's four pricing columns.

    The 6/12/24-hour "predicted" columns are exact forward shifts: the
    residential tariff is deterministic, and day-ahead market prices are
    published a day in advance, so a perfect 24-hour price forecast is the
    realistic assumption in both cases (load and PV still have to be forecast).
    """
    return pd.DataFrame({
        "electricity_pricing": prices,
        "electricity_pricing_predicted_6h": np.roll(prices, -6),
        "electricity_pricing_predicted_12h": np.roll(prices, -12),
        "electricity_pricing_predicted_24h": np.roll(prices, -24),
    })


def dam_prices(n: int, year: int = 2024) -> np.ndarray:
    """Ukrainian day-ahead market prices in UAH/kWh, aligned to the dataset."""
    path = RAW / f"oree_dam_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/fetch_dam_prices.py --year {year}")
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    df = df[~((dates.dt.month == 2) & (dates.dt.day == 29))]     # drop leap day
    hourly = df.iloc[:, 1:].to_numpy(dtype=float).ravel() / 1000  # UAH/MWh -> /kWh
    # One hour is missing each spring (DST transition); carry the previous price
    missing = int(np.isnan(hourly).sum())
    if missing:
        idx = np.where(np.isnan(hourly))[0]
        hourly[idx] = hourly[idx - 1]
    if len(hourly) != n:
        raise ValueError(f"dataset has {n} rows, DAM series has {len(hourly)}")
    aligned = np.roll(hourly, -DATASET_START_INDEX)
    print(f"  DAM {year}: {missing} gap(s) filled, "
          f"{aligned.min():.3f}-{aligned.max():.2f} UAH/kWh, mean {aligned.mean():.2f}")
    return np.round(aligned, 4)


# --------------------------------------------------------------------------
# PV
# --------------------------------------------------------------------------

def kyiv_pv_series(n: int) -> np.ndarray:
    """Kyiv PVGIS year as W/kW AC, aligned to the dataset's August->July start."""
    cache = RAW / "pvgis_kyiv_2019.json"
    if not cache.exists():
        import urllib.request
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PVGIS_URL) as r:  # noqa: S310 (fixed URL)
            cache.write_bytes(r.read())
    hourly = json.loads(cache.read_text())["outputs"]["hourly"]
    power = np.array([h["P"] for h in hourly], dtype=float)  # W per kWp, AC
    if len(power) != 8760:
        raise ValueError(f"expected 8760 hourly values, got {len(power)}")
    local = np.roll(power, KYIV_UTC_OFFSET)          # UTC -> EET
    aligned = np.roll(local, -DATASET_START_INDEX)   # -> 31 July 23:00 start
    if n != len(aligned):
        raise ValueError(f"dataset has {n} rows, PV series has {len(aligned)}")
    return np.round(aligned, 2)


def apply_pv(dst: Path, source: str) -> None:
    if source == "texas":
        return
    building_files = sorted(dst.glob("Building_*.csv"))
    series = None
    for path in building_files:
        df = pd.read_csv(path)
        if series is None:
            series = kyiv_pv_series(len(df))
        df["solar_generation"] = series
        df.to_csv(path, index=False)
    print(f"  PV: Kyiv PVGIS 2019 applied to {len(building_files)} buildings "
          f"({series.sum() / 1000:.0f} kWh/kWp/year)")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build(output_dir: str | Path = "data/citylearn_ua",
          pv: str = "kyiv",
          outages: str = "none") -> Path:
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
    print(f"  pricing: three-zone (default), two-zone, flat — {n} rows each")
    try:
        prices_to_frame(dam_prices(n)).to_csv(dst / "pricing_dam.csv", index=False)
    except FileNotFoundError as exc:
        print(f"  pricing_dam.csv skipped: {exc}")

    apply_pv(dst, pv)

    if outages != "none":
        from make_outage_schedules import apply_outages  # noqa: PLC0415
        apply_outages(dst, scenario=outages)

    return dst


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("output_dir", nargs="?", default="data/citylearn_ua")
    ap.add_argument("--pv", choices=["kyiv", "texas"], default="kyiv",
                    help="solar generation profile (default: kyiv)")
    ap.add_argument("--outages", choices=["none", "scheduled_4x2", "stochastic"],
                    default="none", help="power outage scenario (default: none)")
    args = ap.parse_args()
    out = build(args.output_dir, pv=args.pv, outages=args.outages)
    print(f"CityLearn-UA written to {out}")


if __name__ == "__main__":
    main()
