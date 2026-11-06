"""Re-optimise the hourly rule map for the three-zone tariff and Kyiv PV.

(1+12) evolution strategy over all 24 hourly values, started from the source
map; sigma falls linearly from 0.04 to 0.01 over 16 generations; objective is
the weighted score on the summer window (the tuning window of the benchmark).
The result is deterministic for a given seed.

    python benchmark/tune_rbc_ua.py [output.json]
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

SEED = 42
LAMBDA = 12
GENERATIONS = 16
SIGMA = (0.04, 0.01)


def score(values):
    from bench import WINDOWS, dataset, run
    amap = {h + 1: float(values[h]) for h in range(24)}
    return run(dataset("three_zone"), "map", WINDOWS["summer"], action_map=amap)["score"]


def main():
    from lib.rbc import DEFAULT_ACTION_MAP
    rng = np.random.RandomState(SEED)
    parent = np.array([DEFAULT_ACTION_MAP[h] for h in range(1, 25)])
    best = score(parent)
    evaluations = 1
    print(f"start {best:.4f}", flush=True)
    with ProcessPoolExecutor(int(os.environ.get("BENCH_WORKERS", "6"))) as pool:
        for gen, sigma in enumerate(np.linspace(*SIGMA, GENERATIONS)):
            kids = [np.round(parent + rng.normal(0, sigma, 24), 4) for _ in range(LAMBDA)]
            scores = list(pool.map(score, kids))
            evaluations += LAMBDA
            i = int(np.argmin(scores))
            if scores[i] < best:
                best, parent = scores[i], kids[i]
            print(f"gen {gen + 1:2d} sigma {sigma:.3f} best {best:.4f}", flush=True)
    result = {"seed": SEED, "evaluations": evaluations, "score": best,
              "action_map": {str(h + 1): float(parent[h]) for h in range(24)}}
    print(json.dumps(result))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
