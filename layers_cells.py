from keras import Layer
import tensorflow as tf
import functions
from functions import propagate_brownian_motion

class PropagationCell(Layer):
    # For ref: self.state_size = [n1, n2, 1]  # states = [(batch, n1), (batch, n2), (batch, 1)]
    def __init__(self, units=32, asset_risk_factor_matrix=None, **kwargs):
        super().__init__()
        self.units = units
        self.asset_risk_factor_matrix = asset_risk_factor_matrix
        n_assets, n_rfs = asset_risk_factor_matrix.shape
        self.state_size = [n_assets] # reflecting n_assets, n_rfs
        self.output_size = self.state_size[0]

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation
    # states = [std_normals, assets]
    # next asset val =
    def call(self, inputs, states):
        #std_normals = inputs
        assets = states[0]
        #brownian_drivers = propagate_brownian_motion(inputs, 1., std_normals) # TODO: Change delta_t
        assets = tf.add(assets, tf.matmul(inputs, tf.transpose(self.asset_risk_factor_matrix))) # batch, n_assets + batch, n_rfs x n_rfs, n_assets
        return assets, [assets] # batch, n_assets