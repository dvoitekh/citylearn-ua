"""CityLearn-UA quickstart: no-control baseline vs a simple tariff-aware rule.

Runs a 15-day, 5-building episode twice — once with idle batteries and once
with a hand-written rule adapted to the Ukrainian three-zone tariff (charge
from midday PV surplus and cheap night energy, discharge through both price
peaks) — and reports the electricity import cost in UAH.

Only requires the ``citylearn`` package (same version the dataset was built
with). Runtime: a few seconds.

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from citylearn.citylearn import CityLearnEnv

DATASET = Path(__file__).resolve().parents[1] / "data" / "citylearn_ua" / "schema.json"
DAYS = 15
N_BUILDINGS = 5


def rule_action(hour: int) -> float:
    """Battery action (fraction of capacity per hour) for the clock hour 0..23."""
    if hour >= 23 or hour < 7:
        return 0.05       # gentle night charging at 1.728 UAH/kWh
    if 11 <= hour < 16:
        return 0.18       # recharge from midday PV surplus
    if 8 <= hour < 11:
        return -0.10      # morning price peak (6.48 UAH/kWh)
    if 20 <= hour < 22:
        return -0.30      # evening price peak (6.48 UAH/kWh)
    return 0.0


def run(controlled: bool) -> tuple[float, float]:
    env = CityLearnEnv(
        schema=str(DATASET),
        central_agent=True,
        buildings=[f"Building_{i}" for i in range(1, N_BUILDINGS + 1)],
        simulation_start_time_step=0,
        simulation_end_time_step=24 * DAYS,
    )
    env.reset()
    cost = imports = 0.0
    done = False
    step = 0
    while not done:
        # action computed at step t is applied by CityLearn during hour t+1
        hour = (step + 1) % 24
        a = rule_action(hour) if controlled else 0.0
        _, _, done, truncated, _ = env.step([[a] * N_BUILDINGS])
        done = done or truncated
        price = float(env.buildings[0].pricing.electricity_pricing[env.time_step])
        for b in env.buildings:
            net = float(b.net_electricity_consumption[-1])
            if net > 0:
                cost += net * price
                imports += net
        step += 1
    return cost, imports


if __name__ == "__main__":
    base_cost, base_imp = run(controlled=False)
    rule_cost, rule_imp = run(controlled=True)
    print(f"{'':24s}{'import, kWh':>14s}{'cost, UAH':>12s}")
    print(f"{'no control':24s}{base_imp:14.0f}{base_cost:12.0f}")
    print(f"{'three-zone rule':24s}{rule_imp:14.0f}{rule_cost:12.0f}")
    print(f"\nsavings: {base_cost - rule_cost:.0f} UAH "
          f"({100 * (base_cost - rule_cost) / base_cost:.1f}%) over {DAYS} days, "
          f"{N_BUILDINGS} buildings")
