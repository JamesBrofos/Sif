import numpy as np
from scipy.spatial.distance import cdist
from .abstract_kernel import AbstractKernel


class MaternKernel(AbstractKernel):
    """Matern-5/2 Kernel Class"""
    def cov(self, model_X, model_Y):
        # Compute the squared Euclidean distance between points.
        nX = model_X / self.length_scale
        nY = model_Y / self.length_scale
        dist_sq = cdist(nX, nY, "sqeuclidean")
        dist = np.sqrt(dist_sq)
        K = (1. + np.sqrt(5.)*dist + 5./3.*dist_sq) * np.exp(-np.sqrt(5.)*dist)
        return self.amplitude * K
