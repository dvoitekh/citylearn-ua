"""Persistence forecaster — predicts next 24h = last 24h observed."""

import numpy as np

from .forecast_base import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    """Simplest reasonable forecasting baseline: lag-24 repetition."""

    def __init__(self, n_buildings: int):
        self.n_buildings = n_buildings
        self.load_hist: dict[int, list] = {i: [] for i in range(n_buildings)}
        self.solar_hist: dict[int, list] = {i: [] for i in range(n_buildings)}

    def predict(self, building_idx, current_step, current_hour):
        load_h = self.load_hist[building_idx]
        solar_h = self.solar_hist[building_idx]

        if len(load_h) >= 24:
            load_pred = np.array(load_h[-24:])
            solar_pred = np.array(solar_h[-24:])
        else:
            load_pred = np.ones(24) * (np.mean(load_h) if load_h else 1.0)
            solar_pred = np.zeros(24)

        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        self.load_hist[building_idx].append(load)
        self.solar_hist[building_idx].append(solar)
        if len(self.load_hist[building_idx]) > 168:
            self.load_hist[building_idx] = self.load_hist[building_idx][-168:]
            self.solar_hist[building_idx] = self.solar_hist[building_idx][-168:]

    def reset(self):
        for i in range(self.n_buildings):
            self.load_hist[i] = []
            self.solar_hist[i] = []
