import numpy as np
import scipy.linalg as spla


def multivariate_normal_sampler(mean, cov, n_samples=1):
    """Samples from a multivariate normal distribution with the specified mean
    and covariance.

    Parameters:
        mean (numpy array): Mean vector of the multivariate normal.
        cov (numpy array): Positive definite covariance matrix of the
            multivariate normal. This matrix is square with the size of each
            dimension equal to the number of elements in the `mean` parameter.
        n_samples (int, optional): The number of samples to generate from the
            multivariate normal.
    """
    L = spla.cholesky(cov)
    Z = np.random.normal(size=(n_samples, cov.shape[0]))
    return Z.dot(L) + mean
