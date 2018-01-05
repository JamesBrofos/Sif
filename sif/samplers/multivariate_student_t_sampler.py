import numpy as np
from .multivariate_normal_sampler import multivariate_normal_sampler


def multivariate_student_t_sampler(mean, cov, dof, n_samples=1):
    """Samples form a multivariate Student-t distribution with the specified
    mean, covariance, and degrees of freedom.

    Parameters:
        mean (numpy array): Mean vector of the multivariate Student-t.
        cov (numpy array): Positive definite covariance matrix of the
            multivariate Student-t. This matrix is square with the size of each
            dimension equal to the number of elements in the `mean` parameter.
        dof (float): The degrees-of-freedom of the multivariate Student-t.
        n_samples (int, optional): The number of samples to generate from the
            multivariate normal.
    """
    m = mean.shape[0]
    u = np.random.gamma(dof / 2., 2. / dof, size=(n_samples, 1))
    Y = multivariate_normal_sampler(np.zeros((m, )), cov, n_samples)
    return Y / np.tile(np.sqrt(u), [1, m]) + mean

