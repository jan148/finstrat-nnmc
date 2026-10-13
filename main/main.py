from tensorflow import keras
from tensorflow.keras import layers
from layers_cells import PropagationCell
from losses import max_portfolio_value
from asset_prices_simulation.simulation import *
from results_analysis.results_analysis_utils import *
from plot_utils import *

n_ts_adjusted = n_time_steps - 1
time_grid_adjusted = time_grid [0:-1]

# ------------ 1. Simulate asset prices ------------
std_normals = np.random.normal(loc=0., scale=1., size=(n_paths, n_ts_adjusted, n_stochastic_drivers))
sim_asset_prices_np = simulate(init_asset_vals, std_normals, vol_matrix, time_grid_adjusted, n_paths)
sim_asset_prices_tf = tf.convert_to_tensor(sim_asset_prices_np)

# ------------ 2. Build models ------------
# Models
nn = keras.Sequential()
plotter_model = keras.Sequential()

# Layers
propagation_layer = layers.RNN(cell=PropagationCell(rnn_layer_input_dict=rnn_layer_input_dict), return_sequences=True) # n_batches, n_times, n_assets

nn.add(propagation_layer)

plotter_model.add(propagation_layer)

nn.summary()

nn.compile(optimizer='adam', loss=max_portfolio_value)

nn.fit(sim_asset_prices_tf, tf.zeros([n_paths, n_ts_adjusted, n_assets]), epochs=epochs)

outs = plotter_model.predict(sim_asset_prices_tf)
init = np.broadcast_to(rnn_layer_input_dict["INPUT_init_abs_alloc"], (outs.shape[0], 1, outs.shape[2]))
outs = np.concatenate([init, outs], axis=1)   # (n_paths, n_time_steps + 1, n_assets)
rel_weights = softmax_dim(outs)

# ------------ 3. Result analysis ------------
# Plotting
plot_paths_new(outs, time_grid)
plot_paths_new(rel_weights, time_grid)

result_df = mean_and_variance(outs)
result_df.to_csv('results_out/result.csv')
print(result_df)