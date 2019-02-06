import numpy as np
from .abstract_kernel import AbstractKernel


class BrownianMotionKernel(AbstractKernel):
    """Brownian Motion Kernel Class"""
    def __init__(self, H=1./2, length_scales=np.array([1., ]), amplitude=1.):
        super().__init__(length_scales=length_scales, amplitude=amplitude)
        self.H = H

    def cov(self, model_X, model_Y=None):
        """Implementation of abstract base class method."""
        if model_Y is None:
            model_Y = model_X
        if model_X.shape[1] != 1 or model_Y.shape[1] != 1:
            raise ValueError("Inputs must have only a single dimension, typically denoted time.")
        nX = model_X / self.length_scales
        nY = model_Y / self.length_scales
        if self.H == 1./2:
            C = np.minimum(nX, nY.T)
        else:
            C = np.zeros((nX.shape[0], nY.shape[0]))
            h = 2*self.H
            for i in range(nX.shape[0]):
                for j in range(nY.shape[0]):
                    s, t = nX[i], nY[j]
                    C[i, j] = 0.5*(s**h + t**h - np.abs(t-s)**h)
        return self.amplitude * C



