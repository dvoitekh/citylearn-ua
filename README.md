# CityLearn-UA

**A CityLearn-compatible dataset for benchmarking building energy control under the
Ukrainian residential electricity tariff.**

Real building load and PV profiles from [CityLearn 2022](https://github.com/intelligent-environments-lab/CityLearn)
(Travis County, TX — 17 residential buildings, 8760 hourly steps, 6.4 kWh battery +
5 kW inverter each) combined with the **actual Ukrainian residential time-of-use
tariff** (three-zone and two-zone variants). Drop-in compatible with any code that
runs CityLearn ≥ 2.1: point `schema` at this dataset and every forecaster, MPC, or
RL agent you already have will run against Ukrainian prices.

![Ukrainian residential tariff vs district load and PV](assets/tariff_profile.png)

The evening price peak (20:00–22:00, ×1.5) lands exactly where district load is high
and PV generation is zero — this is the window a battery controller has to win.

## Why this dataset exists

Tariff structure is not a detail — it is the objective function. Controllers tuned
under one time-of-use structure can *lose money* under another: in our benchmark, a
rule schedule Bayesian-optimized for the source (single-evening-peak) tariff charges
batteries straight into the Ukrainian **morning** peak. CityLearn-UA lets you measure
that transferability gap on realistic profiles with a real, currently effective tariff
— and it is, to our knowledge, the first open CityLearn dataset with Ukrainian prices.

## Tariff specification

Residential base price **4.32 UAH/kWh** (in force until 2026-10-31). Zone boundaries
follow the official multi-zone metering rules:

| Variant | File | Night 23:00–07:00 | Peaks 08:00–11:00, 20:00–22:00 | Other hours |
|---|---|---|---|---|
| **Three-zone** (default) | `pricing.csv` | ×0.4 → **1.728** | ×1.5 → **6.48** | 4.32 |
| Two-zone | `pricing_two_zone.csv` | ×0.5 → **2.16** | — | 4.32 |
| Flat (control condition) | `pricing_flat.csv` | 4.32 | 4.32 | 4.32 |

The tariff is deterministic, so the `*_predicted_{6h,12h,24h}` columns are exact
cyclic shifts. To switch variants, overwrite the default:
`cp data/citylearn_ua/pricing_two_zone.csv data/citylearn_ua/pricing.csv`.

## Quickstart

```bash
pip install "citylearn>=2.1" pandas numpy
python examples/quickstart.py
```

Expected output (no-control baseline vs. a simple hand-written three-zone rule,
15 days × 5 buildings):

```
                           import, kWh   cost, UAH
no control                        1424        5375
three-zone rule                   1517        4810

savings: 565 UAH (10.5%) over 15 days, 5 buildings
```

Using the dataset from your own code is one line:

```python
from citylearn.citylearn import CityLearnEnv
env = CityLearnEnv(schema="data/citylearn_ua/schema.json", central_agent=True)
```

## Benchmark results

Electricity import cost over the first 15 days, 5 buildings, three-zone tariff
(controllers from the companion DistrictEMS platform):

![Benchmark results](assets/benchmark.png)

| Controller | Import, kWh | Cost, UAH | CO₂, kg* | Peak-hour import, kWh |
|---|---|---|---|---|
| No control (idle batteries) | 1415 | 5361 | 257.5 | 327 |
| RBC — rules optimized for the source tariff | 1319 | 4986 | 238.9 | 286 |
| RBC-UA — rules re-optimized for this tariff (ES, ~200 evals) | 1294 | 4870 | 238.3 | 233 |
| **MPC — zero re-tuning, prices read as input** | 1294 | **4631** | **232.2** | 234 |

Three findings worth stealing:

1. **Predictive control transfers for free**: −13.6% cost, −28.6% peak-hour import
   vs. no control, with the tariff consumed as plain input data.
2. **Rule schedules do not transfer**: the foreign-tariff rules degrade (they charge
   into the 08:00–11:00 peak that didn't exist in the source tariff); re-optimizing
   the schedule (evolution strategy over the 24 hourly actions) recovers the loss but
   couples the controller to this specific tariff again.
3. **A zero-shot RL policy degrades like the rules do** (+0.040 weighted score vs.
   +0.032 for rules) and would need retraining — hours of compute vs. zero for MPC.

*Caveats: rule maps are optimized on the evaluated window (optimistic upper bound for
rules); the CO₂ column uses the source grid's carbon intensity series and should be
read as a relative indicator only.*

## Rebuilding from scratch

The dataset is fully reproducible from the installed CityLearn package:

```bash
python scripts/build_dataset.py data/citylearn_ua   # ~5 s
python scripts/make_figures.py                      # regenerates assets/
```

## Repository layout

```
data/citylearn_ua/        the dataset (schema, 17 building CSVs, weather,
                          carbon intensity, three pricing variants)
scripts/build_dataset.py  deterministic generator (CityLearn 2022 -> CityLearn-UA)
scripts/make_figures.py   README figures
examples/quickstart.py    baseline vs. simple tariff-aware rule, plain CityLearn
```

## Limitations

- Building load/PV profiles are Texas measurements — climate and consumption habits
  differ from Ukraine; the tariff, not the profiles, is the Ukrainian component.
- Carbon intensity is the source dataset's series (no open hourly Ukrainian data).
- Daylight-saving shifts are not modeled; zones are applied by dataset clock hour.
- Tariff values are effective as of 2026 and set to expire 2026-10-31; the generator
  makes updating coefficients a one-line change.

## License and attribution

MIT (see `LICENSE`). Building profiles, weather, and carbon data originate from the
MIT-licensed [CityLearn](https://github.com/intelligent-environments-lab/CityLearn)
project — if you use this dataset, please also cite CityLearn:

> Vázquez-Canteli J.R., Kämpf J., Henze G., Nagy Z. *CityLearn v1.0: An OpenAI Gym
> environment for demand response with deep reinforcement learning.* BuildSys 2019.

## Citation

See `CITATION.cff`, or:

> Voitekh D. *CityLearn-UA: a CityLearn-compatible dataset for benchmarking building
> energy control under the Ukrainian residential electricity tariff.* 2026.
> https://github.com/dvoitekh/citylearn-ua
