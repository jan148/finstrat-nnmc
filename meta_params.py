import tensorflow as tf

epochs = 5

n_paths = 10
n_times = 20
n_stochastic_drivers = 2
n_assets = 5
asset_risk_factor_matrix = tf.ones((n_assets, n_stochastic_drivers))
x0 = tf.convert_to_tensor([1., 1., 1., 1., 1.])

ddd = tf.shape(asset_risk_factor_matrix)[0].numpy()
ddd = ddd + 9