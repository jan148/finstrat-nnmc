from matplotlib import pyplot as plt
from matplotlib.lines import lineStyles
from configs.config import *
from utils import *

def create_summary_stats(sim_paths):
    pvs = np.sum(sim_paths, axis=-1)
    vars = np.var(pvs, axis=0)
    q_5s = np.percentile(pvs, 5, axis=0)
    q_95s = np.percentile(pvs, 95, axis=0)
    rel_weights = np.apply_along_axis(convert_to_weights, -1, sim_paths)

    results_dict = defaultdict(list)
    results_dict["PV"] = pvs
    results_dict["Variance"] = vars
    results_dict["Absolute Allocations"] = sim_paths
    results_dict["Relative Weights"] = rel_weights
    results_dict["q5"] = q_5s
    results_dict["q95"] = q_95s

    return results_dict

def create_results_dashboard(sim_paths, price_paths, config,  n_paths_in):
    n_paths = sim_paths.shape[0] if n_paths_in is None else n_paths_in
    time_grid = config.time_grid
    res_dict = create_summary_stats(sim_paths)
    pvs, vars, rel_weights = res_dict["PV"], res_dict["Variance"], res_dict["Relative Weights"]

    fig, axs = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(f"Results for {type(config).__name__}")
    axs[0, 0].set(title="Asset prices", ylabel="Price")
    axs[1, 0].set(title="Portfolio value per path", ylabel="Portfolio value")
    axs[1, 1].set(title="Portfolio value variance", ylabel="Variance")
    axs[2, 0].set(title="Absolute allocations", xlabel="Time", ylabel="Allocation")
    axs[2, 1].set(title="Relative weights", xlabel="Time", ylabel="Weight")

    asset_colors = plt.cm.tab10(np.arange(config.n_assets))
    path_colors = plt.cm.viridis(np.linspace(0, 1, n_paths))
    axs[1, 1].plot(time_grid, vars, label='Simulation', color='black') # Variance
    axs[1, 0].plot(time_grid, np.mean(pvs, axis=0), color='black', linestyle='dashdot') # Mean
    for p in range(n_paths):
        axs[1, 0].plot(time_grid, pvs[p], color=path_colors[p])
        for n in range(config.n_assets):
            color = asset_colors[n]
            axs[0, 0].plot(time_grid, price_paths[p, :, n], color=color, alpha=0.5)
            axs[2, 0].plot(time_grid, sim_paths[p, :, n], color=color, alpha=0.5)
            axs[2, 1].plot(time_grid, rel_weights[p, :, n], color=color, alpha=0.5)

    return res_dict, fig, axs

# -------- Deprecated --------
def plot_paths_new(tensor_in, time_grid, n_plot=None):
    tensor = tensor_in if n_plot is None else tensor_in[:n_plot]

    n_paths = tensor.shape[0]
    n_series = tensor.shape[-1]

    fig, subplots = plt.subplots(nrows=1, ncols=2)

    # 1. Plot paths
    for idx in range(n_series):
        color = colors[idx]
        for p in range(n_paths):
            ts = tensor[p, :, idx]
            subplots[0].plot(time_grid, ts, color=color)

    # 2. Plot some summary stats
    sums = np.sum(tensor, axis=2)
    means = np.mean(sums, axis=0)
    vars = np.var(sums, axis=0)

    subplots[1].plot(time_grid, means, label="Mean")
    subplots[1].plot(time_grid, vars, label="Variance")

    plt.show()



