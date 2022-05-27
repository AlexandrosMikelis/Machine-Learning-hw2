import numpy as np

def adam(t, initial_Z, gradient, learning_rate, beta1, beta2, eps=1e-8):

    m = [0.0 for _ in range(len(initial_Z))]
    v = [0.0 for _ in range(len(initial_Z))]

    for i in range(len(initial_Z)):
        m[i] = m[i] * beta1 + (1.0 - beta1) * gradient[i]

        v[i] = v[i] * beta2 + (1.0 - beta2) * gradient[i] ** 2

        mhat = m[i] / (1.0 - beta1**(t+1))
        vhat = v[i] / (1.0 - beta2**(t+1))

        initial_Z[i] = initial_Z[i] - learning_rate * \
            mhat / (np.sqrt(vhat) + eps)


    return initial_Z