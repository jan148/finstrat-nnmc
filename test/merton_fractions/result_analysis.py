import numpy as np
from utils import *
from configs.config_merton_fractions import *
from plot_utils import *

# Predictions
preds = np.load("test_prediction.npy")
weights_actual = np.apply_along_axis(convert_to_weights, 2, preds)

# Targets given by Merton fractions
inverse_cov_assets = np.linalg.inv(MertonFractionsConfig.factor_loading_matrix[1:, 1:]) # Take only asset part
drifts_assets = MertonFractionsConfig.drifts_const[1:]
weights_assets = MertonFractionsConfig.gamma**-1 * inverse_cov_assets @ (drifts_assets - MertonFractionsConfig.r)
weight_ba = 1 - np.sum(weights_assets)
weights_target = np.broadcast_to(np.insert(weights_assets, 0, weight_ba), weights_actual.shape)

plot_paths_new(weights_actual, MertonFractionsConfig.time_grid, 1000)
plot_paths_new(weights_target, MertonFractionsConfig.time_grid, 1000)

