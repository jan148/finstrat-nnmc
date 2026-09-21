from keras import Layer
import tensorflow as tf
import functions
from functions import propagate_brownian_motion

class PropagationCell(Layer):
    # For ref: self.state_size = [n1, n2, 1]  # states = [(batch, n1), (batch, n2), (batch, 1)]
    def __init__(self, units=32, asset_risk_factor_matrix=None, x0=None, **kwargs):
        super().__init__()
        self.units = units
        self.asset_risk_factor_matrix = asset_risk_factor_matrix
        self.initial_value = 1.
        self.x0 = x0
        self.n_assets, self.n_rfs = asset_risk_factor_matrix.shape
        self.state_size = [self.n_assets, self.n_assets] # reflecting n_assets, n_rfs
        self.output_size = self.state_size[0]

        # Weights
        self.w = self.add_weight(shape=(self.n_assets, self.n_assets), trainable=True)

    def get_initial_state(self, batch_size=None):
        alloc0 = tf.ones((batch_size, self.n_assets)) * (self.initial_value / self.n_assets)
        assets0 = tf.broadcast_to(self.x0, (batch_size, self.n_assets))
        return [alloc0, assets0]

    # Defines the computation
    # states = [std_normals, assets]
    # next asset val =
    def call(self, inputs, states):
        alloc, assets = states
        #brownian_drivers = propagate_brownian_motion(inputs, 1., std_normals) # TODO: Change delta_t

        # Propagate assets
        assets = tf.add(assets, tf.matmul(inputs, tf.transpose(self.asset_risk_factor_matrix))) # batch, n_assets + batch, n_rfs x n_rfs, n_assets

        # Determine portfolio value
        total_val = tf.tensordot(alloc, tf.ones((self.n_assets, 1)), axes=1)

        # Rebalance
        rel_alloc = self.rebalance(assets, total_val)
        alloc = total_val * rel_alloc

        return alloc, [alloc, assets] # batch, n_assets; atch, n_assets

    # This takes the assets
    def rebalance(self, assets, total_val):
        weights =  tf.nn.softmax(tf.matmul(assets, self.w), axis=-1) # n_batches, n_assets
        return weights

