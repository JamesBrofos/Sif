import numpy as np
from scipy.spatial.distance import cdist
from .abstract_kernel import AbstractKernel


class SquaredExponentialKernel(AbstractKernel):
    """Squared Exponential Kernel Class"""
    def cov(self, model_X, model_Y):
        # Compute the squared Euclidean distance between points.
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
