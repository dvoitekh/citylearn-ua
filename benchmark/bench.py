"""Benchmark core: one controller on one 15-day window of CityLearn-UA.

Every source of randomness is explicit. The scenario noise of the MPC is set by
``noise_seed``; the rule schedules and the stored SAC policy have no random inputs.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
from citylearn.citylearn import CityLearnEnv

from lib.mpc import MPCController
from lib.persistence import PersistenceForecaster
from lib.rbc import DEFAULT_ACTION_MAP, UA_3ZONE_ACTION_MAP, RBCController

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DERIVED = HERE / "_data"
SAC_MODEL = HERE / "models" / "sac_p12_seed42"
N_BUILDINGS = 5
BUILDINGS = [f"Building_{i}" for i in range(1, N_BUILDINGS + 1)]
NOISE_SEEDS = tuple(range(1, 11))
# Start day in the dataset calendar (row 0 = 31 July, 23:00-24:00)
WINDOWS = {"summer": 0, "autumn": 60, "winter": 150}
WINDOW_DAYS = 15
EXPENSIVE_ABOVE = {"three_zone": 6.4, "dam": 6.0}


def dataset(signal: str) -> str:
    """Schema path of a price signal; derived copies are created on first use."""
    if signal == "three_zone":
        return str(ROOT / "data" / "citylearn_ua" / "schema.json")
    if signal == "source":
        return "citylearn_challenge_2022_phase_all"
    source = ROOT / "data" / "citylearn_ua"
    target = DERIVED / signal
    if not target.exists():
        shutil.copytree(source, target)
        name = {"dam": "pricing_dam.csv", "two_zone": "pricing_two_zone.csv"}[signal]
        shutil.copy(source / name, target / "pricing.csv")
    return str(target / "schema.json")


def _kpis(env):
    k = env.evaluate()
    d = k[k["level"] == "district"]

    def get(name):
        row = d[d["cost_function"] == name]
        v = row["value"].iloc[0] if len(row) else np.nan
        return 1.0 if np.isnan(v) else float(v)

    m = {"cost_component": get("cost_total"), "emissions": get("carbon_emissions_total"),
         "ramping": get("ramping_average"), "load_factor": get("daily_one_minus_load_factor_average"),
         "daily_peak": get("daily_peak_average"), "all_time_peak": get("all_time_peak_average")}
    m["grid"] = (m["ramping"] + m["load_factor"] + m["daily_peak"] + m["all_time_peak"]) / 4
    m["score"] = (m["cost_component"] + m["emissions"] + m["grid"]) / 3
    u = d[d["cost_function"] == "power_outage_normalized_unserved_energy_total"]
    if len(u) and u["value"].notna().any():
        m["unserved"] = float(u["value"].iloc[0])
    return m


def _run_sac(schema, sim_start, sim_end):
    from stable_baselines3 import SAC

    from lib.sb3_wrapper import CityLearnSB3Env
    env = CityLearnSB3Env({"dataset_name": schema, "buildings": BUILDINGS,
                           "sim_start": sim_start, "sim_end": sim_end})
    model = SAC.load(str(SAC_MODEL), device="cpu")
    obs, _ = env.reset()
    while not env.terminated:
        action, _ = model.predict(np.asarray(obs, dtype=np.float32).flatten(), deterministic=True)
        obs, _, _, _, _ = env.step(action.flatten())
    return _kpis(env._base_env.unwrapped)


def run(schema, ctrl, start_day, n_scenarios=3, noise_seed=1, action_map=None, expensive_above=None):
    """Return score components, money and outage metrics of one run."""
    sim_start, sim_end = start_day * 24, (start_day + WINDOW_DAYS) * 24 - 1
    started = time.time()
    if ctrl == "sac":
        out = _run_sac(schema, sim_start, sim_end)
        out.update(controller=ctrl, start_day=start_day, seconds=time.time() - started)
        return out
    if ctrl == "mpc":
        controller = MPCController(forecaster=PersistenceForecaster(N_BUILDINGS), n_buildings=N_BUILDINGS,
                                   sim_start=sim_start, dataset_name=schema, n_scenarios=n_scenarios,
                                   action_smooth_weight=0.1, reopt_interval=8, random_seed=noise_seed)
    else:
        amap = {"none": {h: 1e-12 for h in range(1, 25)}, "rbc": DEFAULT_ACTION_MAP,
                "rbc_ua": UA_3ZONE_ACTION_MAP, "map": action_map}[ctrl]
        controller = RBCController(N_BUILDINGS, action_map=amap, sim_start=sim_start)
    rule = ctrl != "mpc"
    env = CityLearnEnv(schema=schema, central_agent=rule, buildings=BUILDINGS,
                       simulation_start_time_step=sim_start, simulation_end_time_step=sim_end)
    controller.reset()
    obs, _ = env.reset()
    money = base = imp = ref = 0.0
    soc = {}
    step = 0
    while not env.terminated:
        actions = controller.get_action(obs)
        if rule:
            obs, _, _, _, _ = env.step([[actions[0][0]] * N_BUILDINGS])
        else:
            obs, _, _, _, _ = env.step(actions)
        # After a step the time index points at the row the action landed on.
        t = env.time_step
        price = round(float(env.buildings[0].pricing.electricity_pricing[t]), 4)
        socs = []
        for b in env.buildings:
            load = round(float(b.energy_simulation.non_shiftable_load[t]), 3)
            solar = round(float(b.solar_generation[t]), 3)
            net = round(float(b.net_electricity_consumption[t]), 3)
            socs.append(round(float(b.electrical_storage.soc[t]), 3))
            money += max(net, 0) * price
            base += max(load - abs(solar), 0) * price
            if expensive_above is not None and price > expensive_above:
                imp += max(net, 0)
                ref += max(load - abs(solar), 0)
        soc[sim_start + step] = sum(socs) / len(socs)
        step += 1
    out = _kpis(env)
    out.update(controller=ctrl, start_day=start_day, steps=step, seconds=time.time() - started,
               cost_uah=money, no_control_cost_uah=base, cost_change_pct=100 * (money / base - 1), _soc=soc)
    if expensive_above is not None and ref:
        out["expensive_import_change_pct"] = 100 * (imp / ref - 1)
    return out
