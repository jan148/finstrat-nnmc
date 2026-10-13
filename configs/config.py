from collections import defaultdict
import tensorflow as tf
import numpy as np

# Train params
epochs = 5

# Init state
init_asset_vals = [1., 2., 3., 4.]
n_assets = len(init_asset_vals)
init_rel_alloc = [1 / n_assets for val in init_asset_vals]
init_value = 1
simulation_end = 1

init_abs_alloc = np.multiply(init_rel_alloc, init_asset_vals) * init_value
init_portfolio_value = np.sum(init_abs_alloc)

# Model params
n_stochastic_drivers = 2
'''
vol_matrix = np.array([[0.12, 0.45],
              [0.33, 0.07],
              [0.58, 0.21],
              [0.04, 0.39]])
'''
vol_matrix = np.zeros(shape=(n_assets, n_stochastic_drivers))

# MC simulation params
n_paths = 20
n_time_steps = 10
time_grid = np.linspace(0, simulation_end, n_time_steps)

# ------------ Other objects ------------

# For rnn cell
rnn_layer_input_dict = defaultdict(list)
rnn_layer_input_dict["INPUT_init_asset_prices"] = init_asset_vals
rnn_layer_input_dict["INPUT_init_rel_alloc"] = init_rel_alloc
rnn_layer_input_dict["INPUT_init_value"] = init_value
rnn_layer_input_dict["INPUT_init_portfolio_value"] = init_portfolio_value
rnn_layer_input_dict["INPUT_time_grid"] = time_grid
rnn_layer_input_dict["INPUT_asset_risk_factor_matrix"] = vol_matrix
rnn_layer_input_dict["INPUT_init_abs_alloc"] = init_abs_alloc

