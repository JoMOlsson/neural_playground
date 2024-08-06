from math import e
from enum import Enum
import numpy as np


class ActivationFunction(Enum):
    SIGMOID = 1
    RELU = 2
    TANH = 3
    LINEAR = 4


def activate(function: ActivationFunction, z):
    if function == ActivationFunction.SIGMOID:
        return sigmoid(z)
    elif function == ActivationFunction.RELU:
        return relu(z)
    elif function == ActivationFunction.LINEAR:
        return linear(z)
    else:
        raise ValueError


def gradient(function: ActivationFunction, z):
    if function == ActivationFunction.SIGMOID:
        return sigmoid_gradient(z)
    elif function == ActivationFunction.RELU:
        return relu_gradient(z)
    elif function == ActivationFunction.LINEAR:
        return linear_gradient(z)
    else:
        raise ValueError


def sigmoid(z):
    """ Calculates the output of the sigmoid function from the provided z-data

    :param z: (float, int, npArray) Input data
    :return g: Output of sigmoid function
    """
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoid_gradient(z):
    """ Calculates the output of the sigmoid derivative function from the provided z-data

    :param z: (float, int, npArray) Input data
    :return g: Output of sigmoid derivative function
    """
    g = sigmoid(z)
    return g * (1 - g)


def relu(z):
    """ Calculates the output of the ReLU function from the provided z-data

    :param z: (float, int, npArray) Input data
    :return g: Output of ReLU function
    """
    g = np.maximum(0, z)
    return g


def relu_gradient(z):
    """ Calculates the output of the ReLU derivative function from the provided z-data

    :param z: (float, int, npArray) Input data
    :return g: Output of ReLU derivative function
    """
    g = np.where(z > 0, 1, 0)
    return g


def linear(z):
    return z


def linear_gradient(z):
    return np.ones_like(z)
