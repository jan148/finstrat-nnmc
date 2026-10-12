from keras import Layer
from archive.functions import *

"""
inputs = simulated asset prices
state = current absolute allocation weights
"""

class PropagationCell(Layer):
    # For ref: self.state_size = [n1, n2, 1]  # states = [(batch, n1), (batch, n2), (batch, 1)]
    def __init__(self, units=32, rnn_layer_input_dict=None, **kwargs):
        super().__init__()
        self.units = units

        # Init state
        self.init_asset_vals = rnn_layer_input_dict["INPUT_init_asset_prices"]
        self.init_rel_alloc = rnn_layer_input_dict["INPUT_init_rel_alloc"]
        self.init_portfolio_value = rnn_layer_input_dict["INPUT_init_portfolio_value"]
        self.time_grid = rnn_layer_input_dict["INPUT_time_grid"]
        self.asset_risk_factor_matrix = rnn_layer_input_dict["INPUT_asset_risk_factor_matrix"]
        self.init_abs_alloc = rnn_layer_input_dict["INPUT_init_abs_alloc"]

        # Requited
        self.n_assets, self.n_rfs = self.asset_risk_factor_matrix.shape
        self.state_size = [self.n_assets, self.n_assets] # relecting current allocation, previous asset prices
        self.output_size = self.state_size[0]

        # Weights
        self.w = self.add_weight(shape=(self.n_assets, self.n_assets), trainable=True)

    # This will be a bit dirty, since it's converting np -> tf
    def get_initial_state(self, batch_size=None):
        alloc0 = tf.broadcast_to(self.init_abs_alloc, (batch_size, self.n_assets))
        assets0 = tf.broadcast_to(tf.convert_to_tensor(self.init_asset_vals) , (batch_size, self.n_assets))
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

        # Rebalance
        alloc_new_rel = self.rebalance(alloc, inputs) # batch x n_assets
        alloc_new = tf.multiply(alloc_new_rel, portfolio_value)

        return alloc_new, [alloc_new, inputs] # batch, n_assets; atch, n_assets

    # asset prices -> weights over asset prices
    def rebalance(self, alloc, curr_asset_prices):
        weights =  tf.nn.softmax(tf.matmul(curr_asset_prices, self.w), axis=-1) # n_batches, n_assets
        return weights

