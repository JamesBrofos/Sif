import numpy as np
import scipy.linalg as spla
from abc import abstractmethod, abstractproperty
from .abstract_model import AbstractModel


class EllipticalProcess(AbstractModel):
    """Elliptical Process Class"""
    def __init__(self, kernel, noise_level=1e-6, prior_mean=0.):
        """Initialize the parameters of the abstract process class."""
        super().__init__()
        self.kernel = kernel
        self.noise_level = noise_level
        self.prior_mean = prior_mean

    def fit(self, X, y):
        """Implementation of abstract base class method."""
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y.ravel()
        self.y_tilde = self.y - self.prior_mean
        n = self.X.shape[0]
        # Compute the covariance matrix of the observed inputs.
        self.K = self.kernel.cov(self.X, self.X) + self.noise_level * np.eye(n)
        # For a numerically stable algorithm, we use Cholesky decomposition.
        self.L = spla.cholesky(self.K, lower=True)
        self.alpha = spla.cho_solve((self.L, True), self.y_tilde)
        self.beta = self.y_tilde.dot(self.alpha)

    def predict(self, X_pred, diagonal=False):
        """Implementation of abstract base class method."""
        # Compute the cross covariance between training and the requested
        # inference locations. Also compute the covariance matrix of the
        # observed inputs and the covariance at the inference locations.
        X_pred = np.atleast_2d(X_pred)
        K_cross = self.kernel.cov(X_pred, self.X)
        v = spla.solve_triangular(self.L, K_cross.T, lower=True)
        # Posterior inference. Notice that we add a small amount of noise to the
        # diagonal for regulatization purposes.
        mean = K_cross.dot(self.alpha) + self.prior_mean
        if diagonal:
            K_pred = self.kernel.var(X_pred)
            cov = K_pred - np.sum(v**2, axis=0) + 1e-6
        else:
            K_pred = self.kernel.cov(X_pred, X_pred)
            cov = K_pred - v.T.dot(v) + 1e-6 * np.eye(K_pred.shape[0])
        return mean, cov

    @abstractproperty
    def log_likelihood(self):
        """Compute the log-likelihood of the data under the elliptical process
        model with the given hyperparameters of the kernel.
        """
        raise NotImplementedError()
