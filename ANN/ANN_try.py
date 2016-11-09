import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivate_sigmoid(y):
    return y * (1.0 - y)

