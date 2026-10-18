import math

def softmax_dim(x):
    rel_weights = x / x.sum(axis=-1, keepdims=True)
    return rel_weights

def caar(x, gamma):
    return (x**(1 - gamma) - 1) / (1 - gamma) if gamma < 1 else math.log(gamma)

def convert_to_weights(x):
    weights = x / x.sum()
    return weights