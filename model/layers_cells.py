import keras
from keras import Layer
from keras.src.callbacks import Callback
from archive.functions import *

"""
inputs = simulated asset prices
state = current absolute allocation weights
"""

class PropagationCell(Layer):
    def __init__(self, config, **kwargs):
        super().__init__()
        #self.units = units

        # Init state
        self.init_asset_vals = config.init_asset_vals
        self.init_abs_alloc = config.init_abs_alloc

        # Requited
        self.n_assets = self.n_rfs = len(self.init_asset_vals)
        self.state_size = [self.n_assets, self.n_assets] # relecting current allocation, previous asset prices
        self.output_size = self.state_size[0]

        # Create network
        self.inner_model = keras.Sequential()
        self.inner_model.add(keras.Input(shape=(self.n_assets,)))
        for i in range(config.n_hidden_layers):
            self.inner_model.add(keras.layers.Dense(config.n_units_per_hidden_layer, kernel_initializer="zeros"))
        self.inner_model.add(keras.layers.Dense(self.n_assets, kernel_initializer="zeros"))

    # This will be a bit dirty, since it's converting np -> tf
    def get_initial_state(self, batch_size=None):
        alloc0 = tf.broadcast_to(tf.cast(self.init_abs_alloc, self.compute_dtype), (batch_size, self.n_assets))
        assets0 = tf.broadcast_to(tf.cast(self.init_asset_vals, self.compute_dtype), (batch_size, self.n_assets))
        return [alloc0, assets0]

    # Defines the computation
    # states = [std_normals, assets]
    # next asset val =
    def call(self, inputs, states):
        alloc, prev_asset_prices = states

        # Calc returns
        returns = tf.divide(inputs, prev_asset_prices)

        # Get new asset prices & portfolio value
        alloc = tf.multiply(alloc, returns) # batch x
        portfolio_value = tf.reduce_sum(alloc, axis=-1, keepdims=True) # batch x
        alloc_rel = alloc / portfolio_value

        # Rebalance
        alloc_new_rel = self.rebalance(alloc_rel, inputs) # batch x n_assets
        alloc_new = tf.multiply(alloc_new_rel, portfolio_value)

        return alloc_new, [alloc_new, inputs] # batch, n_assets; atch, n_assets

    # Asset prices -> weights over asset prices
    def rebalance(self, alloc_rel, curr_asset_prices):
        raw = self.inner_model(alloc_rel)
        risky = raw[:, 1:]  # risky weights, unconstrained
        bank = 1.0 - tf.reduce_sum(risky, axis=-1, keepdims=True)
        return tf.concat([bank, risky], axis=-1)  # rows sum to 1

class GetWeights(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        for v in self.model.trainable_weights:
            print(f"\n{v.path}, {v.numpy()}")
