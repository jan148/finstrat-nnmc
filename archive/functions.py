import math
import tensorflow as tf

@tf.function
def propagate_brownian_motion(prev_vale, delta_t, std_normals):
    return tf.add(prev_vale, tf.multiply(tf.sqrt(delta_t), std_normals))

@tf.function
def get_gross_returns_and_new_asset_vals(prev_vale, std_normals, delta_t, vol_matrix, drift = 0.01):
    drift_times_asset = tf.multiply(prev_vale, drift) # S_t * r
    drift_inc = tf.multiply(drift_times_asset, delta_t) # S_t * r * dt

    normals = tf.matmul(std_normals, tf.sqrt(delta_t)) # sigma * sqrt(dt)
    vol_times_asset_price = tf.matmul(normals, tf.transpose(vol_matrix)) # sigma * sqrt(dt) * dW(t)
    diff_inc = tf.multiply(prev_vale, vol_times_asset_price)  # S_t * sigma * sqrt(dt) * dW(t)

    asset_prices_new = tf.add(tf.add(prev_vale, drift_inc), diff_inc)
    returns = tf.divide(asset_prices_new, prev_vale)

    return returns, asset_prices_new