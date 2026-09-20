import math
import tensorflow as tf

@tf.function
def propagate_brownian_motion(prev_vale, delta_t, std_normals):
    return tf.add(prev_vale, tf.multiply(tf.sqrt(delta_t), std_normals))