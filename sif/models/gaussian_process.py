import numpy as np
import scipy.linalg as spla
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalTriL


class GaussianProcess:
    """Gaussian Process Class"""
    def __init__(self, kernel, noise_level=1.):
        """Initialize the parameters of the Gaussian process object."""
        self.kernel = kernel
        self.noise_level = noise_level

    def fit(self, X, y):
        """Fit the parameters of the Gaussian process based on the available
        bundles training data.
        """
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y
        n = self.X.shape[0]
        # Compute the covariance matrix of the observed inputs.
        K = self.kernel.cov(self.X, self.X) + self.noise_level * np.eye(n)
        # For a numerically stable algorithm, we use Cholesky decomposition.
        self.L = spla.cholesky(K, lower=True)
        self.alpha = spla.cho_solve((self.L, True), self.y).ravel()

    def predict(self, X_pred):
        """Leverage Bayesian posterior inference to compute the predicted mean
        and variance of a given set of inputs given the available training data.
        Notice that it is necessary to first fit the Gaussian process model
        before posterior inference can be performed.
        """
        # Compute the cross covariance between training and the requested
        # inference locations. Also compute the covariance matrix of the
        # observed inputs and the covariance at the inference locations.
        K_pred = self.kernel.cov(X_pred, X_pred)
        K_cross = self.kernel.cov(X_pred, self.X)
        v = spla.solve_triangular(self.L, K_cross.T, lower=True)
        # Posterior inference. Notice that we add a small amount of noise to the
        # diagonal for regulatization purposes.
        mean = K_cross.dot(self.alpha)
        cov = K_pred - v.T.dot(v) + 1e-6 * np.eye(K_pred.shape[0])
        return mean, cov

    def sample(self, X_pred, n_samples=1):
        """Sample target variables from the predictive posterior distribution of
        the Gaussian process.
        """
        # Bundles hopes sampling algorithm gets better soon <3
        mean, cov = self.predict(X_pred)
        L = spla.cholesky(cov)
        return np.random.normal(size=(n_samples, L.shape[0])).dot(L) + mean

    @property
    def log_likelihood(self):
        """Compute the log-likelihood of the data under the Gaussian process
        model with the given length scales, amplitude, and noise level of the
        kernel.
        """
        return -1. * (
            0.5 * self.y.ravel().dot(self.alpha) +
            np.sum(np.log(np.diag(self.L))) +
            0.5 * self.X.shape[0] * np.log(2.*np.pi)
        )
