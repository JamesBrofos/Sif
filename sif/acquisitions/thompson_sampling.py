import numpy as np
from .abstract_acquisition import AbstractAcquisitionFunction
from ..kernels import compute_kernel_fourier_features
from ..models import BayesianLinearRegression


class ThompsonSampling(AbstractAcquisitionFunction):
    """Thompson Sampling Acquisition Function Class"""
    def __init__(self, models, n_bases):
        """Initialize the parameters of the Thompson sampling acquisition
        function object.
        """
        # Call the initialization method of the base class and set the number of
        # random Fourier features we'd like to assemble.
        super().__init__(models)
        self.n_bases = n_bases

        # Assume a prior mean of zeros and a wide diagonal covariance matrix.
        #This permits very flexible random Fourier feature interpolations.
        l2_reg = 1. / 100
        # Create a list of Bayesian linear regression objects.
        self.W, self.B, self.lrs, self.coef = [], [], [], []
        for i, mod in enumerate(self.models):
            W, B = mod.kernel.sample_spectrum(self.n_bases)
            P = compute_kernel_fourier_features(mod.X, W, B)
            m = BayesianLinearRegression(l2_reg)
            m.fit(P, mod.y)
            self.lrs.append(m)
            self.W.append(W)
            self.B.append(B)
            self.coef.append(m.sample()[0].ravel())

    def evaluate(self, X, integrate=True):
        """Implementation of abstract base class method."""
        m, n = self.n_models, X.shape[0]
        f = np.zeros((m, n))
        for i, mod in enumerate(self.lrs):
            P = compute_kernel_fourier_features(X, self.W[i], self.B[i])
            f[i] = P.dot(self.coef[i])
        if integrate:
            return f.mean(axis=0)
        else:
            return f

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        # So let's say that `x` is 1 x k
        # Then `W` has to be D x k
        # The random Fourier features have to produce a D-vector.
        # The coefficients for the random Fourier features must also be a
        # D-vector.
        m, k = self.n_models, x.shape[1]
        grads = np.zeros((m, k))
        for i, mod in enumerate(self.lrs):
            S = -np.sin(x.dot(self.W[i].T) + self.B[i]).ravel()
            grads[i] = (np.sqrt(2. / self.n_bases) * S * self.W[i].T).dot(self.coef[i])
        return grads.mean(axis=0)
