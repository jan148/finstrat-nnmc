from tensorflow import keras
from tensorflow.keras import layers
from model.layers_cells import PropagationCell
from losses import max_sharpe
from asset_prices_simulation.simulation import *
from results_analysis.results_analysis_utils import *
from plot_utils import *

# ------------ 1. Load simulated test asset prices ------------
train_tf = tf.io.parse_tensor(tf.io.read_file("train.bin"), out_type=tf.float32)
test_tf = tf.io.parse_tensor(tf.io.read_file("test.bin"), out_type=tf.float32)

# ------------ 2. Load models & train ------------
# Models
nn = keras.Sequential()
plotter_model = keras.Sequential()

# Layers
propagation_layer = layers.RNN(cell=PropagationCell(rnn_layer_input_dict=rnn_layer_input_dict), return_sequences=True) # n_batches, n_times, n_assets

nn.add(propagation_layer)
plotter_model.add(propagation_layer)

nn.summary()

nn.compile(optimizer='adam', loss=max_sharpe)

nn.fit(train_tf, tf.zeros([tf.shape(train_tf)[0], n_ts_adjusted, n_assets]), epochs=epochs)

# Summary train-test stats
train_err = nn.evaluate(train_tf, tf.zeros([tf.shape(train_tf)[0], n_ts_adjusted, n_assets]))
test_err = nn.evaluate(test_tf, tf.zeros([tf.shape(test_tf)[0], n_ts_adjusted, n_assets]))
print(f"Train error: {train_err}\tTest error: {test_err}")

outs = plotter_model.predict(train_tf)
init = np.broadcast_to(rnn_layer_input_dict["INPUT_init_abs_alloc"], (outs.shape[0], 1, outs.shape[2]))
outs = np.concatenate([init, outs], axis=1)   # (n_paths, n_time_steps + 1, n_assets)
rel_weights = softmax_dim(outs)

# ------------ 3. Result analysis ------------
# Plotting
plot_paths_new(outs, time_grid)
# plot_paths_new(rel_weights, time_grid)

#result_df = mean_and_variance(outs)
#result_df.to_csv('results_out/result.csv')
#print(result_df)