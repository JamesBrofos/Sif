import numpy as np
from abc import abstractmethod


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
    def __init__(self, length_scales=np.array([1., ]), amplitude=1.):
        """Initialize the parameters of the squared exponential kernel object.
        """
        # Define a variable for the length scales and amplitude.
        self.length_scales = length_scales
        self.amplitude = amplitude

    @abstractmethod
    def grad_input(self, x, Y):
        """Compute the gradient of the kernel with respect to the first input of
        the covariance function and essentially conditioned on the set of
        observed inputs.

        Parameters:
            x (numpy array): A one-dimensional numpy array representing a
                location in the input space. This function computes the gradient
                of the covariance matrix with respect to infinitesimally small
                changes in this input.
            Y (numpy array): A two-dimensional numpy array representing the
                observed inputs.
        Returns:
            A the gradient of the covariance matrix with respect to changes in
                the input argument `x`.

        Raises:
            NotImplementedError: As an abstract method, all classes inheriting
                from the abstract kernel class must implement this method.
        """
        raise NotImplementedError()

