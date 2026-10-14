from keras.src.trainers.data_adapters.array_slicing import train_validation_split
from configs.config import *
from asset_prices_simulation.simulation import *

std_normals = np.random.normal(loc=0., scale=1., size=(n_paths, n_ts_adjusted, n_stochastic_drivers))
sim_asset_prices_np = simulate(init_asset_vals, std_normals, vol_matrix, time_grid_adjusted, n_paths)
sim_asset_prices_tf = tf.convert_to_tensor(sim_asset_prices_np)
sim_asset_prices_tf_train, sim_asset_prices_tf_test = train_validation_split(sim_asset_prices_tf, 0.3)

# Save
tf.io.write_file("train.bin", tf.io.serialize_tensor(sim_asset_prices_tf_train))
tf.io.write_file("test.bin", tf.io.serialize_tensor(sim_asset_prices_tf_test))