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

    def var(self, model_X):
        """This method computes only the variance (not the full covariance
        matrix) of the input. This is easy to compute because most kernels have
        diagonal entries of one (without multiplying by the amplitude of the
        kernel). This method is equivalent to computing

        >>> np.diag(kernel.cov(model_X))

        That is, the diagonal of the full covariance matrix.
        """
        return self.amplitude * np.ones((model_X.shape[0], ))

    @abstractmethod
    def cov(self, model_X, model_Y=None):
        """This method computes the kernel matrix between every row of the first
        input and every other row of the second input. If the second input is
        not provided, then this is just the covariance matrix of the first
        input. Otherwise it is the cross-covariance. Either way, this is the
        full covariance matrix and not just the diagonal.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_spectrum(self, n_bases):
        """Draws samples from the spectrum of the kernel. Note that we must
        return both the bias and the linear coefficients in order for these
        features to remain fixed for further applications (e.g. for inference
        with a test set).

        Parameters:
            n_bases (int): The number of samples (or Fourier bases) of the
                spectrum to generate. The more bases are computed, the higher
                the fidelity of the approximation; however, this makes using the
                bases less useful since they will be slower.

        Raises:
            NotImplementedError: As an abstract method, all classes inheriting
                from the abstract kernel class must implement this method.
        """
        raise NotImplementedError()

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

