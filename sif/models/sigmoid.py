import numpy as np


def sigmoid(z):
    """The sigmoid is a squashing function that compresses the entire real line
    into the unit interval.
    """
    return 1. / (1. + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function with respect to its input."""
    p = sigmoid(z)
    return p * (1. - p)

