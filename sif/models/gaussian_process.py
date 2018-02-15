import numpy as np
import scipy.linalg as spla
from .abstract_process import AbstractProcess
from ..samplers import multivariate_normal_sampler


class GaussianProcess(AbstractProcess):
    """Gaussian Process Class"""
    def __init__(self, kernel, noise_level=1., prior_mean=0.):
        """Initialize the parameters of the Gaussian process object."""
        super().__init__(kernel, noise_level, prior_mean)

    def sample(self, X_pred, n_samples=1):
        """Implementation of abstract base class method."""
        # Bundles hopes sampling algorithm gets better soon <3
        mean, cov = self.predict(X_pred)
        return multivariate_normal_sampler(mean, cov, n_samples)

    def grad_input(self, x):
        """Compute the gradient of the mean function and the standard deviation
        function at the provided input.
        """
        # Compute the gradient of the mean function.
        d_kernel = self.kernel.grad_input(x, self.X)
        d_mean = d_kernel.T.dot(self.alpha)
        # Compute the gradient of the standard deviation function. It is
        # absolutely crucial to note that the predict method returns the
        # variance, not the standard deviation, of the prediction.
        sd = np.sqrt(self.predict(x)[1])
        K_cross = self.kernel.cov(x, self.X)
        M = spla.cho_solve((self.L, True), K_cross.T).ravel()
        d_sd = -d_kernel.T.dot(M) / sd
        return d_mean, d_sd

    @property
    def log_likelihood(self):
        """Implementation of abstract base class property."""
        return -1. * (
            0.5 * self.beta +
            np.sum(np.log(np.diag(self.L))) +
            0.5 * self.X.shape[0] * np.log(2.*np.pi)
        )
