import numpy as np
import scipy.linalg as spla
from abc import abstractmethod, abstractproperty


class AbstractProcess:
    """Abstract Process Class"""
    def __init__(self, kernel, noise_level):
        """Initialize the parameters of the abstract process class."""
        self.kernel = kernel
        self.noise_level = noise_level

    def fit(self, X, y):
        """Fit the parameters of the process based on the available bundles
        training data.
        """
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y.ravel()
        n = self.X.shape[0]
        # Compute the covariance matrix of the observed inputs.
        self.K = self.kernel.cov(self.X, self.X) + self.noise_level * np.eye(n)
        # For a numerically stable algorithm, we use Cholesky decomposition.
        self.L = spla.cholesky(self.K, lower=True)
        self.alpha = spla.cho_solve((self.L, True), self.y)
        self.beta = self.y.dot(self.alpha)

    def predict(self, X_pred):
        """Leverage Bayesian posterior inference to compute the predicted mean
        and variance of a given set of inputs given the available training data.
        Notice that it is necessary to first fit the process model before
        posterior inference can be performed.
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

    @abstractmethod
    def sample(self, X_pred, n_samples=1):
        """Sample target variables from the predictive posterior distribution of
        the process.
        """
        raise NotImplementedError()

    @abstractproperty
    def log_likelihood(self):
        """Compute the log-likelihood of the data under the Gaussian process
        model with the given hyperparameters of the kernel.
        """
        raise NotImplementedError()
