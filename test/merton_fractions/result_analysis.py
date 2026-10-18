import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
from configs.create_config_instances import *
from results_analysis.results_analysis_utils import *

# Sim assets
sim_paths = np.load("sim_assets.npy")

# Predictions
preds = np.load("test_prediction.npy")
weights_actual = np.apply_along_axis(convert_to_weights, 2, preds)

# Targets given by Merton fractions
inverse_cov_assets = np.linalg.inv((merton_config.factor_loading_matrix @ merton_config.factor_loading_matrix.T)[1:, 1:]) # Take only asset part
drifts_assets = merton_config.drifts_const[1:]
weights_assets = merton_config.gamma**-1 * inverse_cov_assets @ (drifts_assets - merton_config.r)
weight_ba = 1 - np.sum(weights_assets)
weights_target = np.broadcast_to(np.insert(weights_assets, 0, weight_ba), weights_actual.shape)

# How close?
mean, var = np.mean(np.abs(weights_actual - weights_target), axis=0), np.var(np.abs(weights_actual - weights_target), axis=0)
means_actual = np.mean(weights_actual, axis=0)
means_target = np.mean(weights_target, axis=0)
for i in range(means_actual.shape[0]):
    for j in range(merton_config.n_assets):
        print(f"Actual: {means_actual[i, j]}, Target: {means_target[i, j]}\t")
    print("\n")

# Dashboard
res_dict, fig, axs = create_results_dashboard(preds, sim_paths, merton_config, n_paths_in=10)
for n, t in enumerate(weights_target[0, 0]):
    axs[2, 1].axhline(t, color=plt.cm.tab10(n), linestyle="--")
df = pd.DataFrame({k: res_dict[k] for k in ["Variance", "q5", "q95"]})
print(df)
plt.show()