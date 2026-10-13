import numpy as np

def get_gross_returns_and_new_asset_vals(prev_vale, std_normals, delta_t, vol_matrix, drift = 0.05):
    drift_inc = np.array(prev_vale) * drift * delta_t # S_t * r * dt

    normals = std_normals * np.sqrt(delta_t) # sigma * sqrt(dt)
    vol_times_asset_price = np.matmul(vol_matrix, normals) # sigma * sqrt(dt) * dW(t)
    diff_inc = np.multiply(prev_vale, vol_times_asset_price)  # S_t * sigma * sqrt(dt) * dW(t)

    asset_prices_new = np.add(np.add(prev_vale, drift_inc), diff_inc).tolist()
    returns = np.divide(asset_prices_new, prev_vale).tolist()

    return returns, asset_prices_new

def simulate(init_vals, std_normals, vol_matrix, time_grid, n_paths):
    time_steps = np.diff(time_grid)

    paths_prices = []

    for p in range(n_paths):
        path = []
        path.append(init_vals)
        for idx, time_step in enumerate(time_steps):
            returns, asset_prices_new = get_gross_returns_and_new_asset_vals(path[-1], std_normals[p][idx], time_step, vol_matrix)
            path.append(asset_prices_new)
        paths_prices.append(path)

    paths_prices_np = np.array(paths_prices)
    return paths_prices_np