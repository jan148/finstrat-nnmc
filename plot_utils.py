import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *

def plot_paths_new(tensor, time_grid, n_plot=None):
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


