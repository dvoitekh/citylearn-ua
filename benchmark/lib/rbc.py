"""
Rule-Based Controller using Bayesian-optimized hourly action map.

Broadcasts same action to all buildings each hour.
"""

import numpy as np

from .base import BaseController

# Hourly rule map adapted to the Ukrainian three-zone residential tariff
# (night 23:00-07:00 = 0.4x, peaks 08:00-11:00 and 20:00-22:00 = 1.5x base) and
# the Kyiv PV year of CityLearn-UA v2.1. Produced by
# scripts/citylearn_ua_benchmark/tune_rbc_ua.py: (1+12) evolution strategy over
# all 24 hourly values, 16 generations, seed 42, started from DEFAULT_ACTION_MAP,
# objective = weighted score on the summer window.
UA_3ZONE_ACTION_MAP = {
    1: -0.0901, 2: -0.0342, 3: -0.0741, 4: -0.0264, 5: -0.0498, 6: 0.0386,
    7: 0.0621, 8: 0.0723, 9: 0.0721, 10: 0.1396, 11: 0.1789, 12: 0.1994,
    13: 0.1705, 14: 0.0367, 15: 0.2766, 16: 0.0285, 17: -0.0446, 18: -0.1595,
    19: -0.1167, 20: -0.1348, 21: -0.1642, 22: -0.1512, 23: -0.1395, 24: 0.0017,
}

# Bayesian-optimized RBC action map (hour -> battery action)
# positive = charge, negative = discharge
DEFAULT_ACTION_MAP = {
    1: -0.061792, 2: -0.040502, 3: -0.001555, 4: -0.089550,
    5: -0.087512, 6: -0.098475, 7: 0.021601, 8: 0.082995,
    9: 0.068180, 10: 0.089157, 11: 0.128125, 12: 0.195553,
    13: 0.188094, 14: 0.179405, 15: 0.195015, 16: 0.024272,
    17: -0.058857, 18: -0.194006, 19: -0.144203, 20: -0.119180,
    21: -0.154150, 22: -0.113250, 23: -0.085495, 24: -0.114294,
}


class RBCController(BaseController):
    """Rule-based controller with hour-indexed action map."""

    def __init__(self, n_buildings: int, action_map: dict[int, float] | None = None,
                 sim_start: int = 0):
        self.n_buildings = n_buildings
        self.action_map = action_map or DEFAULT_ACTION_MAP
        self.sim_start = sim_start
        self.step = 0

    def get_action(self, observations):
        global_step = self.sim_start + self.step
        hour_key = (global_step % 24) + 1  # 1-indexed for action map
        action_value = self.action_map.get(hour_key, 0.0)

        actions = [np.array([action_value], dtype=np.float32) for _ in range(self.n_buildings)]
        self.step += 1
        return actions

    def reset(self):
        self.step = 0

    def explain(self, observations, actions):
        global_step = self.sim_start + self.step - 1
        hour_key = (global_step % 24) + 1
        a = self.action_map.get(hour_key, 0.0)
        direction = "charge" if a > 0.01 else ("discharge" if a < -0.01 else "hold")
        reason = f"hour {hour_key:02d} rule → {direction} {a:+.3f}"
        return [reason] * self.n_buildings

    @property
    def name(self):
        return "RBC"
