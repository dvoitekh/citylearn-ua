"""Base controller interface for building energy management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseController(ABC):
    """
    Base class for all controllers (RBC, MPC, RL, Hybrid).

    Each controller:
    1. Receives observations from the environment
    2. Returns actions (battery charge/discharge) for each building
    3. Can be trained (for RL-based controllers) or configured (for MPC/RBC)
    """

    @abstractmethod
    def get_action(self, observations: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute battery actions for all buildings.

        Args:
            observations: list of observation arrays (one per building)

        Returns:
            actions: list of arrays, one per building (each shape (1,))
        """

    def train(self, config: dict) -> dict[str, Any]:
        """
        Train the controller (for RL-based controllers).

        Args:
            config: training configuration

        Returns:
            dict with training metrics (time, episodes, etc.)
        """
        return {}

    def reset(self) -> None:
        """Reset controller state for a new episode."""

    @abstractmethod
    def explain(self, observations: list[np.ndarray], actions: list[np.ndarray]) -> list[str]:
        """Natural-language description of why each building's action was chosen.

        Part of the mandatory controller contract: a controller class that does not
        implement ``explain`` cannot be instantiated, so every controller on the
        platform can be audited step by step. This body is a reusable baseline
        (direction + magnitude of the action) that subclasses may call via ``super()``.
        """
        reasons = []
        for act in actions:
            a = float(np.asarray(act).flatten()[0])
            if abs(a) < 0.01:
                reasons.append("idle")
            elif a > 0:
                reasons.append(f"charge {a:+.2f}")
            else:
                reasons.append(f"discharge {a:+.2f}")
        return reasons

    @property
    def name(self) -> str:
        return self.__class__.__name__
