# CityLearn-UA reference benchmark

Reproduces every number of the reference results: five buildings, three 15-day windows
(summer 1-15 August, autumn 30 September - 14 October, winter 29 December - 12 January).

| File | Purpose |
|---|---|
| `bench.py` | one controller on one window: score components, money, unserved energy |
| `run_benchmark.py` | all cells and the eight blackout shifts, writes JSON lines |
| `tune_rbc_ua.py` | (1+12) evolution strategy for the RBC-UA schedule, seed 42 |
| `summarize.py` | tables from `results/raw.jsonl` |
| `lib/` | rule-based and scenario MPC controllers, persistence forecast, SB3 wrapper |
| `models/sac_p12_seed42.zip` | stored soft actor-critic policy, trained on the source dataset |
| `results/` | raw results of the published run |

## Randomness

The scenario noise of the MPC comes from `noise_seed`; results are reported over seeds 1-10.
The same seed reproduces the same trajectory. Rule schedules and the stored policy have no
random inputs. The evolution strategy uses seed 42.

## Running

```bash
pip install "citylearn==2.2b0" cvxpy stable-baselines3 pandas numpy
python scripts/build_dataset.py data/citylearn_ua_outage --outages scheduled_4x2
cd benchmark
OMP_NUM_THREADS=1 python run_benchmark.py results/raw.jsonl   # 272 runs, about an hour on 8 cores
python summarize.py
```

`run_benchmark.py` overwrites the given file; use another path to keep the published results.
Derived price datasets are created in `benchmark/_data/`, and `data/` is not modified.
