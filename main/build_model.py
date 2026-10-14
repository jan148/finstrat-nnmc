from tensorflow import keras
from tensorflow.keras import layers
from model.layers_cells import PropagationCell
from losses import max_sharpe
from asset_prices_simulation.simulation import *
from results_analysis.results_analysis_utils import *
from plot_utils import *

# Models
nn = keras.Sequential()
plotter_model = keras.Sequential()

# Layers
propagation_layer = layers.RNN(cell=PropagationCell(rnn_layer_input_dict=rnn_layer_input_dict), return_sequences=True) # n_batches, n_times, n_assets

nn.add(propagation_layer)
plotter_model.add(propagation_layer)

nn.summary()

nn.compile(optimizer='adam', loss=max_sharpe)

# Save
nn.save('model.keras')
plotter_model.summary('plotter.keras')
# nn = keras.models.load_model("model.h5")