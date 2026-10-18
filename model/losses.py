import numpy as np
import tensorflow as tf
from utils import *

'''
y_true = batch_size x time_steps x n_assets
'''

def max_portfolio_value(y_true, y_pred):
    objective = tf.reduce_sum(tf.reduce_sum(y_pred[:, -1], 1))
    return -objective

def max_sharpe(y_true, y_pred):
    pv = tf.reduce_sum(y_pred, axis=-1)                  # (batch, time)
    returns = pv[:, 1:] / pv[:, :-1] - 1.0               # per-step returns
    mean = tf.reduce_mean(returns, axis=-1)
    std = tf.math.reduce_std(returns, axis=-1)
    sharpe = mean / (std + 1e-8)
    return -sharpe                                       # minimize -Sharpe

def get_merton_loss(gamma, lam):
    def merton_fractions(y_true, y_pred):
        w_exp = tf.maximum(tf.reduce_sum(y_pred, axis=-1, keepdims=True), 1e-8)
        w = w_exp[:, -1, 0]
        y_pred_weights = y_pred / w_exp
        if gamma == 1.0:
            u = tf.math.log(w)
        else:
            u = tf.pow(w, 1.0 - gamma) / (1.0 - gamma)
        u = u - lam * tf.reduce_mean(tf.math.reduce_variance(y_pred_weights, axis=1))
        return -u
    return merton_fractions