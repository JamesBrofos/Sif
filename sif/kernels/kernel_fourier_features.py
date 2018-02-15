import numpy as np

def compute_kernel_fourier_features(X, W, B):
    n_bases = len(B)
    return np.sqrt(2. / n_bases) * np.cos(X.dot(W.T) + B)
