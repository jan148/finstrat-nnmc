import numpy as np
import tensorflow as tf

def get_gross_returns_and_new_asset_vals(prev_vale, std_normals, delta_t, vol_matrix, drift):
    drift_inc = np.multiply(np.array(prev_vale), drift) * delta_t # S_t * r * dt

    normals = std_normals * np.sqrt(delta_t) # sigma * sqrt(dt)
    vol_times_asset_price = np.matmul(vol_matrix, normals) # sigma * sqrt(dt) * dW(t)
    diff_inc = np.multiply(prev_vale, vol_times_asset_price)  # S_t * sigma * sqrt(dt) * dW(t)

    asset_prices_new = np.add(np.add(prev_vale, drift_inc), diff_inc).tolist()
    returns = np.divide(asset_prices_new, prev_vale).tolist()

    return returns, asset_prices_new

def simulate(init_vals, vol_matrix, time_grid, n_paths, correlation_matrix=None, drifts=None):
    std_normals = np.random.normal(loc=0., scale=1., size=(n_paths, len(time_grid) - 1, vol_matrix.shape[1]))
    if correlation_matrix is None:
        id_matrix = np.identity(vol_matrix.shape[1])
        correlation_matrix = id_matrix
    sqrt_corr_mat = tf.linalg.sqrtm(tf.convert_to_tensor(correlation_matrix, dtype=tf.float32)).numpy()
    correlated_std_normals = std_normals @ sqrt_corr_mat.T # Last dim: std-normal

    time_steps = np.diff(time_grid)
    paths_prices = []
    for p in range(n_paths):
        path = []
        path.append(init_vals)
        for idx, time_step in enumerate(time_steps):
            drift = drifts[idx]
            returns, asset_prices_new = get_gross_returns_and_new_asset_vals(path[-1], correlated_std_normals[p][idx], time_step, vol_matrix, drift)
            path.append(asset_prices_new)
        paths_prices.append(path)

    paths_prices_np = np.array(paths_prices)
    return paths_prices_np

def simulate_w_config(config):
    init_vals= config.init_asset_vals
    vol_matrix= config.factor_loading_matrix
    time_grid = config.time_grid
    n_paths = config.n_paths
    correlation_matrix = config.correlation_matrix
    drifts = config.drifts

    paths_prices_np = simulate(init_vals, vol_matrix, time_grid, n_paths, correlation_matrix, drifts)
    return paths_prices_np