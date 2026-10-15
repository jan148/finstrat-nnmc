import math

import tensorflow as tf

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
]

def softmax_dim(x):
    rel_weights = x / x.sum(axis=-1, keepdims=True)
    return rel_weights

def caar(x, gamma):
    return (x**(1 - gamma) - 1) / (1 - gamma) if gamma < 1 else math.log(gamma)

def convert_to_weights(x):
    weights = x / x.sum()
    return weights