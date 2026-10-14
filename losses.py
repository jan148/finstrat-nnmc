import tensorflow as tf

'''
y_true = batch_size x time_steps x n_assets
'''

def max_portfolio_value(y_true, y_pred):
    objective = -tf.reduce_sum(tf.reduce_sum(y_pred[:, -1], 1))
    return objective

def max_sharpe(y_true, y_pred):
    pv = tf.reduce_sum(y_pred, axis=-1)                  # (batch, time)
    returns = pv[:, 1:] / pv[:, :-1] - 1.0               # per-step returns
    mean = tf.reduce_mean(returns, axis=-1)
    std = tf.math.reduce_std(returns, axis=-1)
    sharpe = mean / (std + 1e-8)
    return -sharpe                                       # minimize -Sharpe