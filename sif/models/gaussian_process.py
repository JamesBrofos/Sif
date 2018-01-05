import numpy as np
from .abstract_process import AbstractProcess
from ..samplers import multivariate_normal_sampler


class GaussianProcess(AbstractProcess):
    """Gaussian Process Class"""
    def __init__(self, kernel, noise_level=1.):
        """Initialize the parameters of the Gaussian process object."""
        super().__init__(kernel, noise_level)

    def sample(self, X_pred, n_samples=1):
        """Implementation of abstract base class method."""
        # Bundles hopes sampling algorithm gets better soon <3
        mean, cov = self.predict(X_pred)
        return multivariate_normal_sampler(mean, cov, n_samples)

    @property
    def log_likelihood(self):
        """Implementation of abstract base class property."""
        return -1. * (
            0.5 * self.beta +
            np.sum(np.log(np.diag(self.L))) +
            0.5 * self.X.shape[0] * np.log(2.*np.pi)
        )
