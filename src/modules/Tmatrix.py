import numpy as np

#Q2 T
def Tmatrix(N_cuts):
    diag_ones = np.eye(N_cuts, N_cuts)
    zeros = np.zeros(shape=(784-N_cuts, N_cuts))

    T_transposed = [*diag_ones, *zeros]
    T = np.transpose(T_transposed)
    return T

def TAvgMatrix():
    T = np.zeros(shape=(49,784))
    starting = 0
    for i in range(49):
        for j in range(starting, starting + 5*28, 28):
            for k in range(4):
                T[i][j+k] = 1/16
        starting = starting + 4
    return T