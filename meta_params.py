import tensorflow as tf

n_paths = 10
n_times = 20
n_stochastic_drivers = 2
n_assets = 5
asset_risk_factor_matrix = tf.ones((n_assets, n_stochastic_drivers))

ddd = tf.shape(asset_risk_factor_matrix)[0].numpy()
ddd = ddd + 9