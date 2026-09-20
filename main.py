import tensorflow as tf
from charset_normalizer import models
from tensorflow import keras
from tensorflow.keras import layers
import layers_cells
from layers_cells import PropagationCell
from meta_params import *

paths = tf.random.normal([n_paths, n_times, n_stochastic_drivers])

nn = keras.Sequential()
nn.add(layers.RNN(cell=PropagationCell(asset_risk_factor_matrix=asset_risk_factor_matrix)))

nn.summary()

#nn.compile(optimizer='adam', loss='mse')