from tensorflow import keras
from tensorflow.keras import layers
from model.layers_cells import PropagationCell

def build_prop_model_from_config(config):
    nn = keras.Sequential()
    propagation_layer = layers.RNN(cell=PropagationCell(config=config), return_sequences=True) # n_batches, n_times, n_assets
    nn.add(propagation_layer)
    nn.summary()
    return nn