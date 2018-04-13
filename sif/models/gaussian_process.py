import numpy as np
import scipy.linalg as spla
from .elliptical_process import EllipticalProcess
from ..samplers import multivariate_normal_sampler


class GaussianProcess(EllipticalProcess):
    """Gaussian Process Class"""
    def sample(self, X_pred, n_samples=1, target=False):
        """Implementation of abstract base class method."""
        # Bundles hopes sampling algorithm gets better soon <3
        X_pred = np.atleast_2d(X_pred)
        mean, cov = self.predict(X_pred)
        if target:
            cov += self.noise_level * np.eye(X_pred.shape[0])
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
