import numpy as np
from .abstract_acquisition import AbstractAcquisitionFunction
from ..models import GaussianProcess, BayesianLinearRegression
from ..kernels import rff


class ThompsonSampling(AbstractAcquisitionFunction):
    """Thompson Sampling Acquisition Function Class"""
    def __init__(self, models):
        """Initialize the parameters of the Thompson sampling acquisition
        function object.
        """
        # Call the initialization method of the base class and set the number of
        # random Fourier features we'd like to assemble.
        super().__init__(models)
        if len(self.models) > 1:
            raise ValueError("Using more than one model with the Thompson sampling acquisition function is invalid.")
