import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x,a=-1):
    return (1 / (1 + np.exp(a*x)))

