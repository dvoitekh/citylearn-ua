"""Power-outage scenarios for CityLearn-UA.

Two mechanisms, both supported by CityLearn 2.2b0:

``scheduled_4x2``
    Deterministic rolling-blackout schedule in the style of the Ukrainian
    winter 2022-2023 queues: repeating 4 hours off / 4 hours on during the
    heating season (November-March), synchronous across the district — which
    is realistic, since a district is switched as one feeder queue. Written as
    a ``power_outage`` column (0/1) in every ``Building_X.csv`` and enabled per
    building in ``schema.json`` (``simulate_power_outage: true``,
    ``stochastic_power_outage: false``).

``stochastic``
    CityLearn's ``ReliabilityMetricsPowerOutage`` (SAIFI/CAIDI), i.e. random
    fault-driven outages rather than planned queues. Enabled purely through
    ``schema.json``; a distinct random seed per building makes outages
    independent. Note: the packaged model silently drops an outage that lands
    on day index 0 (``citylearn/power_outage.py:154``).

The dataset's calendar runs August->July: row 0 is 31 July hour 24, so month
boundaries are taken from the ``month`` column of the building CSVs rather
than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HEATING_SEASON = {11, 12, 1, 2, 3}
HOURS_OFF = 4
HOURS_ON = 4
SAIFI = 1.436          # interruptions per customer per year (EIA reference)
CAIDI = 331.2          # average interruption duration, minutes


def scheduled_signal(months: pd.Series) -> np.ndarray:
    """4h-off / 4h-on rolling blackout during the heating season."""
    period = HOURS_OFF + HOURS_ON
    in_season = months.isin(HEATING_SEASON).to_numpy()
    # phase counted within the season so the pattern starts at each season entry
    signal = np.zeros(len(months), dtype=int)
    phase = 0
    for i, active in enumerate(in_season):
        if not active:
            phase = 0
            continue
        signal[i] = 1 if phase % period < HOURS_OFF else 0
        phase += 1
    return signal


def _schema_outage_block(scenario: str, seed: int) -> dict:
    if scenario == "stochastic":
        return {
            "simulate_power_outage": True,
            "stochastic_power_outage": True,
            "stochastic_power_outage_model": {
                "type": "citylearn.power_outage.ReliabilityMetricsPowerOutage",
                "attributes": {"random_seed": seed, "saifi": SAIFI, "caidi": CAIDI},
            },
        }
    return {"simulate_power_outage": True, "stochastic_power_outage": False}


def apply_outages(dataset_dir: str | Path, scenario: str = "scheduled_4x2") -> None:
    dst = Path(dataset_dir)
    schema_path = dst / "schema.json"
    schema = json.loads(schema_path.read_text())

    building_files = sorted(dst.glob("Building_*.csv"))
    hours_off_total = 0
    for i, path in enumerate(building_files):
        df = pd.read_csv(path)
        if scenario == "scheduled_4x2":
            signal = scheduled_signal(df["month"])
            df["power_outage"] = signal
            df.to_csv(path, index=False)
            hours_off_total = int(signal.sum())
        elif "power_outage" in df.columns:      # stochastic: drop stale column
            df = df.drop(columns=["power_outage"])
            df.to_csv(path, index=False)

    for i, name in enumerate(sorted(schema["buildings"])):
        schema["buildings"][name]["power_outage"] = _schema_outage_block(
            scenario, seed=65647 + i)
    schema.setdefault("observations", {}).setdefault(
        "power_outage", {"active": True, "shared_in_central_agent": True})
    schema["observations"]["power_outage"]["active"] = True
    schema_path.write_text(json.dumps(schema, indent=2))

    if scenario == "scheduled_4x2":
        print(f"  outages: scheduled_4x2 — {hours_off_total} outage hours "
              f"({hours_off_total / 87.6:.1f}% of the year), synchronous district-wide")
    else:
        print(f"  outages: stochastic (SAIFI={SAIFI}, CAIDI={CAIDI} min), "
              f"independent per building")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset_dir", nargs="?", default="data/citylearn_ua")
    ap.add_argument("--scenario", choices=["scheduled_4x2", "stochastic"],
                    default="scheduled_4x2")
    args = ap.parse_args()
    apply_outages(args.dataset_dir, args.scenario)
