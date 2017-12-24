import numpy as np


class AbstractKernel:
    """Abstract Kernel Class

    Kernel functions are leveraged by probabilistic processes for computing the
    covariance between inputs. Common examples of kernel functions include the
    Matern kernel and the squared exponential; kernels can also be computed as
    direct sums of other kernels, and noise can be introduced into the kernel
    in order to reflect stochastic outputs.

    This class implements several utilities for working with kernel functions
    for application to Bayesian optimization. In particular, the class supports
    sampling from the kernel parameters, computing the covariance between two
    matrix-valued inputs, as well as functions for computing the gradient of the
    kernel with respect to both kernel parameters and vector inputs.
    """
    def __init__(self, length_scale=np.array([1., ]), amplitude=1.):
        """Initialize the parameters of the squared exponential kernel object.
        """
        # Define a variable for the length scales and amplitude.
        self.length_scale = length_scale
        self.amplitude = amplitude


