from keras import Layer
import tensorflow as tf
import functions
from functions import propagate_brownian_motion
from meta_params import asset_risk_factor_matrix

class PropagationCell(Layer):
    # For ref: self.state_size = [n1, n2, 1]  # states = [(batch, n1), (batch, n2), (batch, 1)]
    def __init__(self, units=32, asset_risk_factor_matrix=None, **kwargs):
        super().__init__()
        self.units = units
        self.asset_risk_factor_matrix = asset_risk_factor_matrix
        self.state_size = [tf.shape(asset_risk_factor_matrix)[0].numpy(), tf.shape(asset_risk_factor_matrix)[0].numpy()] # reflecting n_assets, n_rfs

    # Create the state of the layer (weights)
    def build(self, input_shape):
        pass

    # Defines the computation
    # states = [std_normals, assets]
    # next asset val =
    def call(self, inputs, states):
        std_normals = inputs
        assets, brownians = states
        brownian_drivers = propagate_brownian_motion(inputs, 1., std_normals) # TODO: Change delta_t
        assets = tf.multiply(self.asset_risk_factor_matrix, brownian_drivers)
        return assets, [brownian_drivers, assets]