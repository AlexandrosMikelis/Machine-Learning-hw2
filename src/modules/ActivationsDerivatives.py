import numpy as np
def sigmoid(x,a=-1):
    return (1 / (1 + np.exp(a*x)))

def ReLU_prime(x):

    for i in range(len(x)):

        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x

def sigmoid_prime(x,a=-1):
    return a * (sigmoid(x) * (1 - sigmoid(x)))