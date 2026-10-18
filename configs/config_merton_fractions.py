from collections import defaultdict
import numpy as np
from configs.config import BaseConfig
from model.losses import get_merton_loss

class MertonFractionsConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Model & train params
        self.epochs = 10
        self.n_hidden_layers = 0
        self.n_units_per_hidden_layer = 2**3
        self.gamma = 0.2
        self.lam = 0
        self.loss_func = get_merton_loss(self.gamma, self.lam)

        # Init state
        self.init_asset_vals = [1., 2., 3., 4.]
        self.n_assets = len(self.init_asset_vals)
        self.init_rel_alloc = np.full(self.n_assets, 1 / self.n_assets)
        self.init_value = 1

        self.init_abs_alloc = np.multiply(self.init_rel_alloc, self.init_asset_vals) * self.init_value
        self.init_portfolio_value = np.sum(self.init_abs_alloc)

        # MC simulation params
        self.simulation_end = 1
        self.n_paths = 10000
        self.n_time_steps = 251 * self.simulation_end
        self.time_grid = np.linspace(0, self.simulation_end, self.n_time_steps + 1)

        # Model params
        self.r = 0.02
        self.drifts_const = self.r + np.array([0, 0.01,  0.04,  0.06])
        self.drifts = np.broadcast_to(self.drifts_const, shape=(self.n_time_steps, self.n_assets))
        self.volas = np.array([0.0, 0.2, 0.3, 0.5])
        self.correlation_matrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
        self.factor_loading_matrix =  np.diag(self.volas) @ np.linalg.cholesky(self.correlation_matrix) # Cholesky decomposition of variance matrix
        self.n_stochastic_drivers = self.factor_loading_matrix.shape[1]