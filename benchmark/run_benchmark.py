"""Run the whole CityLearn-UA benchmark and write raw results as JSON lines.

Cells: three-zone and day-ahead prices on three windows, two-zone on the summer
window; the blackout scenario on the winter window for all eight phase shifts of
the 4 h / 4 h cycle. The MPC runs once per noise seed in ``bench.NOISE_SEEDS``
with 3 scenarios (the reference) and, for sensitivity, with 5.

    python benchmark/run_benchmark.py benchmark/results/raw.jsonl
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

CELLS = [("three_zone", "summer"), ("three_zone", "autumn"), ("three_zone", "winter"),
         ("dam", "summer"), ("dam", "autumn"), ("dam", "winter"), ("two_zone", "summer")]


def shifted_dataset(shift, tmp):
    """Copy of the outage dataset with the blackout schedule delayed by ``shift`` hours."""
    from bench import ROOT
    source = ROOT / "data" / "citylearn_ua_outage"
    if shift == 0:
        return source
    target = tmp / f"outage_shift{shift}"
    if not target.exists():
        shutil.copytree(source, target)
        for csv in target.glob("Building_*.csv"):
            df = pd.read_csv(csv)
            df["power_outage"] = np.roll(df["power_outage"].to_numpy(), shift)
            df.to_csv(csv, index=False)
    return target


def job(spec):
    from bench import EXPENSIVE_ABOVE, WINDOWS, WINDOW_DAYS, dataset, run
    kind = spec["kind"]
    if kind == "cell":
        r = run(dataset(spec["signal"]), spec["ctrl"], WINDOWS[spec["window"]],
                n_scenarios=spec.get("S", 3), noise_seed=spec.get("noise", 1),
                expensive_above=EXPENSIVE_ABOVE.get(spec["signal"]))
        r.pop("_soc")
    else:
        root = Path(spec["dir"])
        start = WINDOWS["winter"]
        r = run(str(root / "schema.json"), spec["ctrl"], start,
                n_scenarios=spec.get("S", 3), noise_seed=spec.get("noise", 1))
        soc = r.pop("_soc")
        outage = pd.read_csv(root / "Building_1.csv")["power_outage"].to_numpy()
        first, last = start * 24, (start + WINDOW_DAYS) * 24 - 1
        starts = [t for t in range(first + 1, last) if outage[t] == 1 and outage[t - 1] == 0]
        if soc:
            r["soc_before_outage"] = float(np.mean([soc[t - 1] for t in starts if t - 1 in soc]))
    r.update({k: v for k, v in spec.items() if k != "dir"})
    return r


def main():
    from bench import NOISE_SEEDS
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_tmp_shifts"
    tmp.mkdir(exist_ok=True)
    specs = []
    for signal, window in CELLS:
        for ctrl in ("none", "rbc", "rbc_ua", "sac"):
            specs.append({"kind": "cell", "signal": signal, "window": window, "ctrl": ctrl})
        for noise in NOISE_SEEDS:
            for s in (3, 5):
                specs.append({"kind": "cell", "signal": signal, "window": window, "ctrl": "mpc",
                              "noise": noise, "S": s})
    for shift in range(8):
        d = str(shifted_dataset(shift, tmp))
        for ctrl in ("none", "rbc", "rbc_ua"):
            specs.append({"kind": "outage", "shift": shift, "dir": d, "ctrl": ctrl})
        for noise in NOISE_SEEDS:
            specs.append({"kind": "outage", "shift": shift, "dir": d, "ctrl": "mpc", "noise": noise, "S": 3})
    print(len(specs), "runs", flush=True)
    with ProcessPoolExecutor(int(os.environ.get("BENCH_WORKERS", "7"))) as pool, open(out, "w") as f:
        futures = [pool.submit(job, s) for s in specs]
        for i, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            if i % 25 == 0:
                print(i, flush=True)
    shutil.rmtree(tmp)
    print("done", flush=True)


if __name__ == "__main__":
    main()
