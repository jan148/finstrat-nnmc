from tensorflow import keras
from tensorflow.keras import layers
from layers_cells import PropagationCell
from losses import max_portfolio_value
from asset_prices_simulation.simulation import *
from results_analysis.results_analysis_utils import *

# ------------ 1. Simulate asset prices ------------
std_normals = np.random.normal(loc=0., scale=1., size=(n_paths, n_time_steps, n_stochastic_drivers))
sim_asset_prices_np = simulate(init_asset_vals, std_normals, vol_matrix, time_grid, n_paths)
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

nn.fit(sim_asset_prices_tf, tf.zeros([n_paths, n_time_steps, n_assets]), epochs=epochs)

outs = plotter_model.predict(sim_asset_prices_tf)

# Plotting
#plot(outs)
ddd = 0

# ------------ 3. Result analysis ------------
result_df = mean_and_variance(outs)
print(result_df)