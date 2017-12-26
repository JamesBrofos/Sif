import numpy as np
import scipy.linalg as spla
from scipy.special import gammaln
from .abstract_process import AbstractProcess


class StudentTProcess(AbstractProcess):
    """Student-t Process Class"""
    def __init__(self, kernel, noise_level, nu):
        super().__init__(kernel, noise_level)
        self.nu = nu

    def sample(self, X_pred, n_samples=1):
        """Implementation of abstract base class method."""
        n = self.X.shape[0]
        eta = self.nu + n
        mean, cov = self.predict(X_pred)
        L = spla.cholesky(cov)
        Z = np.random.normal(size=(n_samples, L.shape[0])).dot(L)
        g = np.tile(
            np.random.gamma(eta / 2., 2. / eta, n_samples),
            (cov.shape[0], 1)
        ).T
        return Z / np.sqrt(g) + mean

    def predict(self, X_pred):
        """Extension of abstract base class method."""
        mean, cov = super().predict(X_pred)
        n = self.X.shape[0]
        cov *= (self.nu + self.beta - 2.) / (self.nu + n - 2.)
        return mean, cov

    # @property
    # def log_likelihood(self):
    #     """Implementation of abstract base class property."""
    #     n = self.X.shape[0]
    #     a = np.sum(np.log(np.diag(self.L)))
    #     b = (
    #         gammaln((self.nu + n) / 2.) -
    #         (n / 2.) * np.log(np.pi(self.nu - 2.)) -
    #         gammaln(self.nu / 2.)
    #     )
    #     c = -((self.nu + n) / 2.) * np.log(
    #         1. + self.y.ravel().dot(self.alpha) / (self.nu - 2.)
    #     )
    #     return a + b + c
