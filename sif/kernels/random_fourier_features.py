import numpy as np

def rff(X, W, B):
    n_bases = len(B)
    return np.sqrt(2. / n_bases) * np.cos(X.dot(W.T) + B)

def grad_rff(x, W, B):
    S = -np.sin(x.dot(W.T) + B).ravel()
    return np.sqrt(2. / self.n_bases) * S * W.T
