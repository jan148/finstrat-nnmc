from keras.src.trainers.data_adapters.array_slicing import train_validation_split
from configs.config_merton_fractions import *
from asset_prices_simulation.simulation import *
from model.layers_cells import *
from configs.create_config_instances import *
from model.build_model import *

# ------------ 1. Load simulated test asset prices ------------
asset_prices_sim = simulate_w_config(merton_config)
sim_asset_prices_tf = tf.convert_to_tensor(asset_prices_sim)
train_asset_prices, test_asset_prices = train_validation_split(sim_asset_prices_tf, 0.3)

# ------------ 2. Compile model & train model & evaluate ------------
nn = build_prop_model_from_config(merton_config)

# Compile
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
nn.compile(optimizer=optimizer, loss=merton_config.loss_func)

# Fit
nn.fit(train_asset_prices
       , tf.zeros([tf.shape(train_asset_prices)[0]
       , merton_config.n_time_steps, merton_config.n_assets])
       , epochs=merton_config.epochs
       , callbacks=[GetWeights()]
       , batch_size=2**6)

# Predict test
test_predictions = nn.predict(test_asset_prices)

# Evaluate
train_err = nn.evaluate(train_asset_prices, tf.zeros([tf.shape(train_asset_prices)[0], merton_config.n_time_steps, merton_config.n_assets]))
test_err = nn.evaluate(test_asset_prices, tf.zeros([tf.shape(test_asset_prices)[0], merton_config.n_time_steps, merton_config.n_assets]))
print(f"Train error: {train_err}\tTest error: {test_err}")

# Save ts
np.save("sim_assets.npy", asset_prices_sim)
np.save("test_prediction.npy", test_predictions)