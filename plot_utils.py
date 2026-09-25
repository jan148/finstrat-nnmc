import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot(matrix: tf.Tensor, alpha: float = 0.3, title: str = "Paths",
         max_cols: int = 3, max_paths: int | None = None) -> None:
    """Plot a (n_paths, n_timesteps, n_assets) tensor.

    One subplot per asset; inside each subplot every path is one line over time.
    """
    data = np.asarray(matrix)  # works for tf tensors and numpy arrays
    if data.ndim != 3:
        raise ValueError(f"Expected shape (n_paths, n_timesteps, n_assets), got {data.shape}")

    n_paths, n_steps, n_assets = data.shape
    if max_paths is not None and n_paths > max_paths:   # optional: plot only a subset
        data = data[:max_paths]
    t = np.arange(n_steps)

    n_cols = min(n_assets, max_cols)
    n_rows = int(np.ceil(n_assets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             sharex=True, squeeze=False)
    axes = axes.ravel()

    for a in range(n_assets):
        ax = axes[a]
        ax.plot(t, data[:, :, a].T, color="tab:blue", alpha=alpha, linewidth=0.8)  # one line per path
        ax.set_title(f"Asset {a}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    for ax in axes[n_assets:]:      # hide unused subplots in the grid
        ax.set_visible(False)

    fig.suptitle(f"{title} ({n_paths} paths)")
    plt.tight_layout()
    plt.show()

def plot_paths(paths, time_grid, n_plot=10, asset_names=None, random=False,
               seed=None, alpha=0.6, lw=1.0, ax=None, title=None):
    """
    Plot simulated paths for multiple assets in a single figure.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, n_times, n_assets)
    time_grid : array-like, shape (n_times,)
    n_plot : int, number of paths to plot per asset
    asset_names : list of str, optional (defaults to "Asset 0", "Asset 1", ...)
    random : bool, pick paths at random instead of the first n_plot
    seed : int, optional RNG seed used when random=True
    alpha, lw : line transparency and width
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    paths = np.asarray(paths)
    time_grid = np.asarray(time_grid)

    if paths.ndim != 3:
        raise ValueError(f"paths must be 3D (n_paths, n_times, n_assets), got shape {paths.shape}")
    n_paths, n_times, n_assets = paths.shape
    if time_grid.shape[0] != n_times:
        raise ValueError(f"time_grid length {time_grid.shape[0]} != n_times {n_times}")

    n_plot = min(n_plot, n_paths)
    if random:
        idx = np.random.default_rng(seed).choice(n_paths, size=n_plot, replace=False)
    else:
        idx = np.arange(n_plot)

    if asset_names is None:
        asset_names = [f"Asset {j}" for j in range(n_assets)]
    elif len(asset_names) != n_assets:
        raise ValueError("asset_names must have length n_assets")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap("tab10" if n_assets <= 10 else "tab20")
    colors = [cmap(j % cmap.N) for j in range(n_assets)]

    for j in range(n_assets):
        # paths[idx, :, j] has shape (n_plot, n_times); transpose so each column is one path
        lines = ax.plot(time_grid, paths[idx, :, j].T, color=colors[j], alpha=alpha, lw=lw)
        lines[0].set_label(asset_names[j])  # one legend entry per asset

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title or f"{n_plot} sample paths per asset")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()