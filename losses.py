import tensorflow as tf

'''
y_true = batch_size x time_steps x n_assets
'''

def max_portfolio_value(y_true, y_pred):
    objective = -tf.reduce_sum(tf.reduce_sum(y_pred[:, -1], 1))
    return objective

