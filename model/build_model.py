from tensorflow import keras
from tensorflow.keras import layers
from model.layers_cells import PropagationCell
from configs.config_merton_fractions import *

# Models
nn = keras.Sequential()
plotter_model = keras.Sequential()

# Layers
propagation_layer = layers.RNN(cell=PropagationCell(config=MertonFractionsConfig), return_sequences=True) # n_batches, n_times, n_assets

nn.add(propagation_layer)
plotter_model.add(propagation_layer)

nn.summary()
#plotter_model.summary('plotter.keras')

# Save
# nn.save('model.keras')
# nn = keras.models.load_model("model.h5")