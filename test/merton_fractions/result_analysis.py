import numpy as np
from utils import *
from configs.config_merton_fractions import *
from plot_utils import *

# Predictions
preds = np.load("test_prediction.npy")
weights_actual = np.apply_along_axis(convert_to_weights, 2, preds)

# Targets given by Merton fractions
r = MertonFractionsConfig.drifts_const[0]
weights = []
for asset_idx in range(MertonFractionsConfig.n_assets):
    mu = MertonFractionsConfig.drifts_const[asset_idx]
    var = MertonFractionsConfig.factor_loading_matrix[asset_idx, asset_idx]
    optimal_alloc = (mu - r) / (var * MertonFractionsConfig.gamma)
    weights.append(optimal_alloc)
weights[0] = 1 - sum(weights[1:])
weights_target = np.broadcast_to(weights, weights_actual.shape)

plot_paths_new(weights_actual, MertonFractionsConfig.time_grid)
plot_paths_new(weights_target, MertonFractionsConfig.time_grid)

