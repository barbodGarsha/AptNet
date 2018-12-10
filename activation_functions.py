import math
import numpy as np
SIGMOID = 0


def activation_f_calculate(activation_function, x):
    if activation_function == SIGMOID:
        return sigmoid(x)


def activation_f_calculate_array(activation_function, x):
    if activation_function == SIGMOID:
        return sigmoid_array(x)


# ----------------------------------------------------------------------------------------

# Activation function
def sigmoid(x):
    x = np.clip(x, -500, 500)
    res = 1 / (1 + math.exp(-x))
    return res


def sigmoid_array(x):
    x2 = []
    for a in x:
        x2.append(sigmoid(a))
    return x2


# Derivatives
def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_p_array(x):
    x2 = []
    for a in x:
        x2.append(sigmoid_p(a))
    return x2

