from configs.config import *
from plot_utils import *
from simulation import simulate

std_normals = np.random.normal(loc=0., scale=1., size=(n_paths, n_time_steps, n_stochastic_drivers))
asset_prices_new = simulate(init_asset_vals, std_normals, vol_matrix, time_grid, n_paths)
plot_paths_new(asset_prices_new, time_grid)