import numpy as np
from scipy.spatial.distance import cdist
from .abstract_kernel import AbstractKernel


class SquaredExponentialKernel(AbstractKernel):
    """Squared Exponential Kernel Class"""
    def cov(self, model_X, model_Y=None):
        # Compute the squared Euclidean distance between points.
        if model_Y is None:
            model_Y = model_X
        nX = model_X / self.length_scales
        nY = model_Y / self.length_scales
        dist_sq = cdist(nX, nY, "sqeuclidean")
        return self.amplitude * np.exp(-0.5 * dist_sq)

    def grad_input(self, x, Y):
        """Implementation of abstract base class method."""
        d_dist = -(x - Y) / (self.length_scales ** 2)
        d_kernel = self.cov(np.atleast_2d(x), Y).T
        grad = d_dist * d_kernel
        return grad

    def sample_spectrum(self, n_bases):
        k = len(self.length_scales)
        B = np.random.uniform(0., 2.*np.pi, size=(n_bases, ))
        W = np.random.normal(size=(n_bases, k)) / self.length_scales
        return W, B
