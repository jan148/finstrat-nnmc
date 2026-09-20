from layers_cells import *

n_paths = 10
n_times = 20
n_stochastic_drivers = 5

paths = tf.random.normal([n_paths, n_times, n_stochastic_drivers])
one_path = paths[1, :, :]
inputs = one_path[0, :]
states = one_path[1, :]

cell = PropagationCell()
a = cell.call(inputs, [states])

ddd = 0
