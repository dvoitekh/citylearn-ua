"""
Online Rolling-Horizon MPC Controller.

Solves 24h rolling horizon LP with CVXPY/CLARABEL.
Pluggable forecaster for load/solar predictions.
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from citylearn.data import DataSet

from .base import BaseController
from .forecast_base import BaseForecaster

# LGB per-horizon load MAPE (h=1..24), measured on Phase 2
LGB_NOISE_PROFILE = np.array([
    0.547, 0.678, 0.693, 0.629, 0.573, 0.548, 0.668, 0.673, 0.728,
    0.725, 0.761, 0.784, 0.804, 0.941, 1.072, 1.161, 0.887, 0.731,
    0.646, 0.568, 0.479, 0.485, 0.499, 0.584,
])


class MPCController(BaseController):
    """Rolling-horizon MPC with pluggable forecaster."""

    def __init__(
        self,
        forecaster: BaseForecaster,
        n_buildings: int,
        n_scenarios: int = 5,
        sim_start: int = 0,
        action_smooth_weight: float = 0.1,
        reopt_interval: int = 8,
        peak_penalty_weight: float = 0.0,
        horizon_discount: float = 1.0,
        noise_profile: np.ndarray | None = None,
        horizon: int = 24,
        random_seed: int = 2022,
        dataset_name: str = "citylearn_challenge_2022_phase_all",
    ):
        self.forecaster = forecaster
        self.n_buildings = n_buildings
        self.n_scenarios = n_scenarios
        self.sim_start = sim_start
        self.action_smooth_weight = action_smooth_weight
        self.reopt_interval = reopt_interval
        self.peak_penalty_weight = peak_penalty_weight
        self.horizon_discount = horizon_discount
        self.noise_profile = noise_profile
        self.horizon = horizon
        self.historical_peak = 0.0
        self.rng = np.random.RandomState(random_seed)

        # Battery parameters
        self.capacity = 6.4
        self.max_power = 5.0
        self.eta = 0.912
        self.max_act = self.max_power / self.capacity  # ~0.78

        # Price/carbon data (loaded from dataset)
        self.price = None
        self.carbon = None
        self.dataset_name = dataset_name

        # Current plan and state
        self.plan = np.zeros((n_buildings, 24))
        self.last_action = np.zeros(n_buildings)
        self.global_step = 0

    def reset(self):
        # Named CityLearn dataset → resolve via DataSet; a local path
        # (e.g. "data/citylearn_ua/schema.json" or a directory) → use it directly.
        from pathlib import Path
        ds = Path(self.dataset_name)
        if ds.suffix == ".json":
            root_dir = str(ds.parent)
        elif ds.is_dir():
            root_dir = str(ds)
        else:
            schema = DataSet.get_schema(self.dataset_name)
            root_dir = schema["root_directory"]

        self.price = pd.read_csv(f"{root_dir}/pricing.csv")["electricity_pricing"].values
        self.carbon = pd.read_csv(f"{root_dir}/carbon_intensity.csv")["carbon_intensity"].values

        self.plan = np.zeros((self.n_buildings, 24))
        self.last_action = np.zeros(self.n_buildings)
        self.global_step = 0
        self.historical_peak = 0.0
        self.forecaster.reset()

    def get_action(self, observations):
        N = len(observations)
        hour = int(observations[0][2]) % 24

        # Update forecaster and track district peak
        district_now = 0.0
        for b in range(N):
            load = observations[b][20]
            solar = observations[b][21]
            self.forecaster.update(b, load, solar, observation=observations[b])
            district_now += (load - solar) / self.capacity
        self.historical_peak = max(self.historical_peak, district_now)

        # Re-optimize at configured interval (after warmup)
        should_reopt = self.global_step % self.reopt_interval == 0

        if should_reopt and self.global_step >= 24:
            load_pred = np.zeros((N, 24))
            solar_pred = np.zeros((N, 24))
            for b in range(N):
                load_pred[b], solar_pred[b] = self.forecaster.predict(b, self.global_step, hour)

            g = self.sim_start + self.global_step
            price_24 = self._get_future_data(self.price, g, 0.25)
            carbon_24 = self._get_future_data(self.carbon, g, 0.4)

            init_soc = [observations[b][22] for b in range(N)]

            result = self._solve_mpc(
                init_soc, load_pred, solar_pred, price_24, carbon_24,
                prev_action=self.last_action, historical_peak=self.historical_peak,
            )
            if result is not None:
                self.plan = result

        # Extract action from current plan
        step_in_plan = self.global_step % self.reopt_interval
        actions = []
        for b in range(N):
            action = self.plan[b, step_in_plan] if step_in_plan < 24 else 0.0
            action = np.clip(action, -self.max_act, self.max_act)
            self.last_action[b] = action
            actions.append(np.array([action], dtype=np.float32))

        self.global_step += 1
        return actions

    def _get_future_data(self, data: np.ndarray, g: int, default: float) -> np.ndarray:
        if g + 25 <= len(data):
            result = data[g + 1:g + 25]
        else:
            result = np.pad(
                data[g + 1:], (0, max(0, 24 - len(data) + g + 1)), constant_values=default
            )
        if len(result) < 24:
            result = np.pad(result, (0, 24 - len(result)), constant_values=default)
        return result

    def _solve_mpc(self, init_soc, load_pred, solar_pred, price_24, carbon_24,
                   prev_action=None, historical_peak=0.0):
        N = self.n_buildings
        T = self.horizon
        S = self.n_scenarios
        eta = self.eta

        load_pred = load_pred[:, :T]
        solar_pred = solar_pred[:, :T]
        price_24 = price_24[:T]
        carbon_24 = carbon_24[:T]

        net_load_base = (load_pred - solar_pred) / self.capacity

        noise_std_h = self.noise_profile[:T] if self.noise_profile is not None else np.full(T, 0.3)

        net_load_scenarios = np.zeros((S, N, T))
        for s in range(S):
            noise = self.rng.normal(0, 1, (N, T)) * noise_std_h[np.newaxis, :]
            net_load_scenarios[s] = net_load_base * (1 + noise)

        if self.horizon_discount < 1.0:
            tw = np.array([self.horizon_discount ** t for t in range(T)])
        else:
            tw = np.ones(T)

        mean_scenario = net_load_scenarios.mean(axis=0)
        district_mean = mean_scenario.sum(axis=0)
        price_base = np.sum(np.maximum(district_mean, 0) * price_24) + 0.01
        emission_base = np.sum(np.maximum(mean_scenario, 0).sum(axis=0) * carbon_24) + 0.01
        ramping_base = np.sum(np.abs(np.diff(district_mean))) + 0.01

        x = cp.Variable((N, T), nonneg=True)
        y = cp.Variable((N, T), nonpos=True)
        constraints = [x <= self.max_act, y >= -self.max_act]

        obj = 0

        for s in range(S):
            L = net_load_scenarios[s]
            soc = cp.Variable((N, T))
            constraints += [soc >= 0, soc <= 1]

            for i in range(N):
                constraints.append(soc[i, 0] == init_soc[i] + x[i, 0] * eta + y[i, 0] / eta)
                for t in range(1, T):
                    constraints.append(
                        soc[i, t] == soc[i, t - 1] + x[i, t] * eta + y[i, t] / eta
                    )

            for t in range(T):
                district_t = cp.sum(L[:, t] + x[:, t] + y[:, t])
                obj += tw[t] * price_24[t] * cp.pos(district_t) / price_base

            for t in range(T):
                for b in range(N):
                    net_b = L[b, t] + x[b, t] + y[b, t]
                    obj += tw[t] * carbon_24[t] * cp.pos(net_b) / emission_base

            for t in range(T - 1):
                d_t = cp.sum(L[:, t] + x[:, t] + y[:, t])
                d_t1 = cp.sum(L[:, t + 1] + x[:, t + 1] + y[:, t + 1])
                obj += 0.5 * tw[t] * cp.abs(d_t1 - d_t) / ramping_base

            district_vars = [cp.sum(L[:, t] + x[:, t] + y[:, t]) for t in range(T)]
            peak = cp.Variable()
            for t in range(T):
                constraints.append(peak >= district_vars[t])
            avg = cp.sum(cp.hstack(district_vars)) / T
            obj += 0.5 * (peak - avg) / (np.max(district_mean) - np.mean(district_mean) + 0.01)

            if self.peak_penalty_weight > 0 and historical_peak > 0:
                peak_excess = cp.pos(peak - historical_peak)
                normalize_base = np.max(district_mean) + 0.01
                obj += self.peak_penalty_weight * peak_excess / normalize_base

        obj = obj / S

        if self.action_smooth_weight > 0:
            for b in range(N):
                for t in range(T - 1):
                    obj += self.action_smooth_weight * cp.abs(
                        (x[b, t + 1] + y[b, t + 1]) - (x[b, t] + y[b, t])
                    )
                if prev_action is not None:
                    obj += self.action_smooth_weight * cp.abs(
                        (x[b, 0] + y[b, 0]) - prev_action[b]
                    )

        problem = cp.Problem(cp.Minimize(obj), constraints)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                result = x.value + y.value
                if T < 24:
                    result = np.pad(result, ((0, 0), (0, 24 - T)), constant_values=0)
                return result
        except Exception:
            pass

        return None

    def explain(self, observations, actions):
        g = self.sim_start + max(0, self.global_step - 1)
        price_now = float(self.price[g]) if self.price is not None and g < len(self.price) else 0.0
        carbon_now = float(self.carbon[g]) if self.carbon is not None and g < len(self.carbon) else 0.0
        reasons = []
        for b, act in enumerate(actions):
            a = float(np.asarray(act).flatten()[0])
            soc = float(observations[b][22]) if len(observations[b]) > 22 else 0.5
            load = float(observations[b][20]) if len(observations[b]) > 20 else 0.0
            solar = float(observations[b][21]) if len(observations[b]) > 21 else 0.0
            net = load - solar
            if a > 0.01:
                reason = f"charge {a:+.2f} (price={price_now:.3f}, SoC={soc:.2f}, net={net:+.2f}kW)"
            elif a < -0.01:
                reason = f"discharge {a:+.2f} (price={price_now:.3f}, carbon={carbon_now:.2f}, SoC={soc:.2f})"
            else:
                reason = f"hold (SoC={soc:.2f})"
            reasons.append(reason)
        return reasons

    @property
    def name(self):
        return "MPC"
