"""
CityLearn <-> Stable-Baselines3 bridge.

Two wrappers:
- CityLearnSB3Env: centralized (all buildings in one obs/action vector)
- PerBuildingSB3Env: per-building (fixed obs -> 1 action, building-agnostic)
"""

import gymnasium as gym
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from gymnasium import spaces

# Default observation features for RL
OPTIMAL_FEATURES = [
    "electrical_storage_soc",
    "net_electricity_consumption",
    "solar_generation",
    "electricity_pricing",
    "carbon_intensity",
    "hour",
    "month",
    "outdoor_dry_bulb_temperature_predicted_6h",
    "diffuse_solar_irradiance_predicted_6h",
]


class CityLearnSB3Env(gym.Env):
    """
    Centralized wrapper: all buildings concatenated into one obs/action vector.

    Obs/action dimensions depend on building count, so a model trained
    on 5 buildings cannot evaluate on 13 buildings.
    """

    def __init__(self, phase_config: dict, reward_fn=None, active_observations=None):
        super().__init__()
        self.reward_fn = reward_fn
        self.active_observations = active_observations or OPTIMAL_FEATURES

        self._base_env = CityLearnEnv(
            schema=phase_config.get("dataset_name", "citylearn_challenge_2022_phase_all"),
            central_agent=True,
            buildings=phase_config["buildings"],
            simulation_start_time_step=phase_config["sim_start"],
            simulation_end_time_step=phase_config["sim_end"],
            active_observations=self.active_observations,
        )

        wrapped = NormalizedObservationWrapper(self._base_env)
        self._env = StableBaselines3Wrapper(wrapped)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def unwrapped_citylearn(self):
        return self._base_env

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        if self.reward_fn is not None:
            self.reward_fn.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self.reward_fn is not None:
            reward = self.reward_fn.calculate(self._base_env)
        return obs, reward, terminated, truncated, info

    @property
    def terminated(self):
        return self._base_env.terminated


class PerBuildingSB3Env(gym.Env):
    """
    Per-building wrapper for building-agnostic RL.

    Each SB3 step() = one full CityLearn timestep:
    1. Receive action for a randomly-sampled building
    2. Apply the SAME action to ALL buildings (parameter sharing)
    3. Step CityLearn once with all actions
    4. Return a randomly-sampled building's next observation

    Obs: (n_features,) for one building (fixed-size)
    Action: (1,) battery action
    """

    def __init__(self, phase_config: dict, reward_fn=None, active_observations=None,
                 normalize_reward=False):
        super().__init__()
        self.reward_fn = reward_fn
        self.active_observations = active_observations or OPTIMAL_FEATURES
        self.normalize_reward = normalize_reward

        self._base_env = CityLearnEnv(
            schema=phase_config.get("dataset_name", "citylearn_challenge_2022_phase_all"),
            central_agent=False,
            buildings=phase_config["buildings"],
            simulation_start_time_step=phase_config["sim_start"],
            simulation_end_time_step=phase_config["sim_end"],
            active_observations=self.active_observations,
        )

        self._norm_env = NormalizedObservationWrapper(self._base_env)
        self.n_buildings = len(self._base_env.buildings)
        self._rng = np.random.RandomState(42)

        # Determine per-building obs size
        norm_obs_space = self._norm_env.observation_space
        if isinstance(norm_obs_space, list):
            self.n_features = norm_obs_space[0].shape[0]
        else:
            self.n_features = norm_obs_space.shape[0] // self.n_buildings

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_features,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self._current_obs = None
        self._model = None

    def set_model(self, model):
        """Enable per-building inference: model predicts for each building individually."""
        self._model = model

    @property
    def unwrapped_citylearn(self):
        return self._base_env

    @property
    def terminated(self):
        return self._base_env.terminated

    def reset(self, **kwargs):
        obs, info = self._norm_env.reset(**kwargs)
        if self.reward_fn is not None:
            self.reward_fn.reset()
        self._current_obs = obs
        b = self._rng.randint(self.n_buildings)
        return self._to_array(obs[b]), info

    def step(self, action):
        target_b = self._rng.randint(self.n_buildings)
        action_val = np.array(action, dtype=np.float32).flatten()

        if self._model is not None and self._current_obs is not None:
            actions = []
            for b in range(self.n_buildings):
                if b == target_b:
                    actions.append(action_val.copy())
                else:
                    obs_b = self._to_array(self._current_obs[b])
                    a, _ = self._model.predict(obs_b, deterministic=False)
                    actions.append(np.array(a, dtype=np.float32).flatten())
        else:
            actions = [action_val.copy() for _ in range(self.n_buildings)]

        obs, rewards, terminated, truncated, info = self._norm_env.step(actions)
        self._current_obs = obs

        if self.reward_fn is not None:
            reward = self.reward_fn.calculate(self._base_env)
        else:
            if isinstance(rewards, list):
                reward = sum(float(r[0]) if hasattr(r, "__len__") else float(r) for r in rewards)
            else:
                reward = float(rewards)

        if self.normalize_reward:
            reward = reward / self.n_buildings

        next_obs = self._to_array(obs[target_b])
        return next_obs, float(reward), bool(terminated), bool(truncated), info

    def _to_array(self, obs):
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        return np.array(obs, dtype=np.float32)
