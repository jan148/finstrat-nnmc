from collections import defaultdict
import numpy as np
from model.losses import get_merton_loss

class MertonFractionsConfig:
    # Train params
    epochs = 10
    n_hidden_layers = 0
    n_units_per_hidden_layer = 2**3
    gamma = 0.2
    lam = 0
    loss_func = get_merton_loss(gamma, lam)

    # Init state
    init_asset_vals = [1., 2., 3., 4.]
    n_assets = len(init_asset_vals)
    init_rel_alloc = np.full(n_assets, 1 / n_assets)
    init_value = 1
    simulation_end = 1

    # MC simulation params
    n_paths = 10000
    n_time_steps = 252 * simulation_end
    time_grid = np.linspace(0, simulation_end, n_time_steps + 1)
    n_ts_adjusted = n_time_steps
    time_grid_adjusted = time_grid[0:-1]

    init_abs_alloc = np.multiply(init_rel_alloc, init_asset_vals) * init_value
    init_portfolio_value = np.sum(init_abs_alloc)

    # Model params
    r = 0.02
    drifts_const = np.array([r, r + 0.01, r + 0.04, r + 0.06])
    drifts = np.broadcast_to(drifts_const, shape=(n_time_steps, n_assets))
    volas = np.array([0.0, 0.2, 0.3, 0.5])
    correlation_matrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
    factor_loading_matrix =  np.diag(volas) @ np.linalg.cholesky(correlation_matrix) # Cholesky decomposition of variance matrix
    n_stochastic_drivers = factor_loading_matrix.shape[1]