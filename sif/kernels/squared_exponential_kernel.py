import numpy as np
from scipy.spatial.distance import cdist
from .abstract_kernel import AbstractKernel


class SquaredExponentialKernel(AbstractKernel):
    """Squared Exponential Kernel Class"""
    def cov(self, model_X, model_Y):
        # Compute the squared Euclidean distance between points.
        nX = model_X / self.length_scale
        nY = model_Y / self.length_scale
        dist_sq = cdist(nX, nY, "sqeuclidean")
        return self.amplitude * np.exp(-dist_sq)


