"""Base class for 24-hour ahead forecasters."""

from abc import ABC, abstractmethod

import numpy as np


class BaseForecaster(ABC):
    """Base class for 24-hour ahead forecasters."""

    @abstractmethod
    def predict(self, building_idx: int, current_step: int, current_hour: int
                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict load and solar for next 24 hours.

        Args:
            building_idx: building index
            current_step: global simulation step
            current_hour: hour of day (0-23)

        Returns:
            load_pred: array (24,) predicted load in kW
            solar_pred: array (24,) predicted solar in kW
        """

    @abstractmethod
    def update(self, building_idx: int, load: float, solar: float,
               observation: np.ndarray | None = None) -> None:
        """
        Update forecaster with new observation.

        Args:
            building_idx: building index
            load: observed load in kW
            solar: observed solar in kW
            observation: full CityLearn observation (optional)
        """

    def reset(self) -> None:
        """Reset forecaster state."""
