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