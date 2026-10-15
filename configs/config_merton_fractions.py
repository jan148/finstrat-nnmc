from collections import defaultdict
import numpy as np
from configs.config import n_time_steps

class MertonFractionsConfig:
    # Train params
    epochs = 50

    # Init state
    init_asset_vals = [1., 1., 1., 1.]
    n_assets = len(init_asset_vals)
    init_rel_alloc = np.full(n_assets, 1 / n_assets)
    init_value = 1
    simulation_end = 1

    init_abs_alloc = np.multiply(init_rel_alloc, init_asset_vals) * init_value
    init_portfolio_value = np.sum(init_abs_alloc)

    # Model params
    r = 0.02
    drifts_const = np.array([r, r + 0.2, r + 0.4, r - 5.6])
    drifts = np.broadcast_to(drifts_const,shape=(n_time_steps - 1, n_assets))
    factor_loading_matrix = np.array([[0, 0, 0, 0],
                                      [0, 0.05, 0, 0],
                                      [0, 0, 0.3, 0],
                                      [0, 0, 0, 0.5]])
    correlation_matrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
    n_stochastic_drivers = factor_loading_matrix.shape[1]
    gamma = 0.5

    # MC simulation params
    n_paths = 20
    n_time_steps = 10
    time_grid = np.linspace(0, simulation_end, n_time_steps)
    n_ts_adjusted = n_time_steps - 1
    time_grid_adjusted = time_grid [0:-1]

    # ------------ Other objects ------------
    # For rnn cell
    rnn_layer_input_dict = defaultdict(list)
    rnn_layer_input_dict["INPUT_init_asset_prices"] = init_asset_vals
    rnn_layer_input_dict["INPUT_init_rel_alloc"] = init_rel_alloc
    rnn_layer_input_dict["INPUT_init_value"] = init_value
    rnn_layer_input_dict["INPUT_init_portfolio_value"] = init_portfolio_value
    rnn_layer_input_dict["INPUT_time_grid"] = time_grid
    rnn_layer_input_dict["INPUT_asset_risk_factor_matrix"] = factor_loading_matrix
    rnn_layer_input_dict["INPUT_init_abs_alloc"] = init_abs_alloc

