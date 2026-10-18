from abc import ABC
from collections import defaultdict
import tensorflow as tf
import numpy as np

class BaseConfig(ABC):
    def __init__(self):
        # Model & train params
        self.epochs = 10
        self.n_hidden_layers = 5
        self.n_units_per_hidden_layer = 2**3
        self.loss_func = None

        # Init state
        self.init_asset_vals = None
        self.n_assets = None
        self.init_rel_alloc = None
        self.init_value = None

        self.init_abs_alloc = None
        self.init_portfolio_value = None

        # MC simulation params
        self.simulation_end = 1
        self.n_paths = 10000
        self.n_time_steps = 251 * self.simulation_end
        self.time_grid = np.linspace(0, self.simulation_end, self.n_time_steps + 1)

        # Model params
        self.drifts = None
        self.factor_loading_matrix = None
        self.n_stochastic_drivers = None