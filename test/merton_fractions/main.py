from keras.src.trainers.data_adapters.array_slicing import train_validation_split
from configs.config_merton_fractions import *
from asset_prices_simulation.simulation import *

n_ts_adjusted = MertonFractionsConfig.n_ts_adjusted
n_assets = MertonFractionsConfig.n_assets
epochs = MertonFractionsConfig.epochs

# ------------ 1. Load simulated test asset prices ------------
asset_prices_sim = simulate_w_config(MertonFractionsConfig)
sim_asset_prices_tf = tf.convert_to_tensor(asset_prices_sim)
train_asset_prices, test_asset_prices = train_validation_split(sim_asset_prices_tf, 0.3)

# ------------ 2. Compile model & train model & evaluate ------------
from model.build_model import *

# Compile
nn.compile(optimizer='adam', loss=MertonFractionsConfig.loss_func)

# Fit
nn.fit(train_asset_prices, tf.zeros([tf.shape(train_asset_prices)[0], n_ts_adjusted, n_assets]), epochs=epochs)

# Predict test
test_predictions = nn.predict(test_asset_prices)
np.save("test_prediction.npy", test_predictions)

# Evaluate
train_err = nn.evaluate(train_asset_prices, tf.zeros([tf.shape(train_asset_prices)[0], n_ts_adjusted, n_assets]))
test_err = nn.evaluate(test_asset_prices, tf.zeros([tf.shape(test_asset_prices)[0], n_ts_adjusted, n_assets]))
print(f"Train error: {train_err}\tTest error: {test_err}")