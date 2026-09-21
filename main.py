import tensorflow as tf
from charset_normalizer import models
from tensorflow import keras
from tensorflow.keras import layers
import layers_cells
from layers_cells import PropagationCell
from meta_params import *
from plot_utils import *

paths = tf.random.normal([n_paths, n_times, n_stochastic_drivers])

# Models
nn = keras.Sequential()
plotter_model = keras.Sequential()

# Layers
propagation_layer = layers.RNN(cell=PropagationCell(asset_risk_factor_matrix=asset_risk_factor_matrix), return_sequences=True) # n_batches, n_times, n_assets

nn.add(propagation_layer)
nn.add(layers.Dense(16, activation='relu'))

plotter_model.add(propagation_layer)

nn.summary()

nn.compile(optimizer='adam', loss='mse')

nn.fit(paths, tf.zeros([n_paths, n_times, 16]), epochs=epochs)

outs = plotter_model.predict(paths)

plot(outs)
ddd = 0