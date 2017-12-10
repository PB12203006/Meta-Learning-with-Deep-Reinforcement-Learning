import numpy as np
from scipy.special import expit

#Return a D dimensional parameter vector
def generate_parameters(D):
    r = np.random.random()
    coefficients = []
    for d in range(D):
        if r < 0.3:
            coefficients.append(np.random.random() * 6 -3)
        elif r < 0.6:
            coefficients.append(np.random.laplace())
        else:
            coefficients.append(np.random.normal())
    return np.array(coefficients)

def generate_inputs(N, D):
    return np.random.normal(size=(N, D))

def distort_X(X):
    N, D = X.shape
    distortion = np.ones(shape=(N,D))
    distortion += np.random.normal(scale=0.1, size=(N, D))
    distorted_X = X * distortion
    return distorted_X

def generate_true_basis(X):
    X_true = np.array(X)
    N, D = X.shape
    for d in range(D):
        if np.random.random() < 0.1:
            r = np.random.random()
            if r < 0.2:
                X_true[:,d] = np.square(X_true[:,d])
            elif r < 0.4:
                X_true[:,d] = np.sin(X_true[:,d])
            elif r < 0.6:
                X_true[:,d] = X_true[:,d] ** 3
            elif r < 0.8:
                X_true[:,d] = np.log(np.abs(X_true[:,d]))
            else:
                X_true[:,d] = np.abs(X_true[:,d])
    return X_true

#generate a N*D feature set and N labels
def generate_data_set(N, D):
    import time
    np.random.seed(seed=int(time.time()))
    X_init = generate_inputs(N, D)
    X = distort_X(X_init)
    X_true = generate_true_basis(X)
    parameters = generate_parameters(D)
    probas = expit(X_true.dot(parameters))
    y = np.array([np.random.random() > probas[n] for n in range(N)])
    return X, y, probas



