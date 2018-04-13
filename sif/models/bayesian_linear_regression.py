import numpy as np
import scipy.linalg as spla
from .generalized_linear_model import GeneralizedLinearModel
from ..samplers import multivariate_normal_sampler


class BayesianLinearRegression(GeneralizedLinearModel):
    """Bayesian Linear Regression Class"""
    def __init__(self, l2_reg, prior_a=1., prior_b=1.):
        """Initialize the parameters of the Bayesian linear regression object.
        """
        super().__init__(l2_reg)
        self.prior_a = prior_a
        self.prior_b = prior_b

    def fit(self, X, y):
        """Implementation of abstract base class method."""
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y.ravel()
        # Produce the prior covariance matrix over the linear coefficients.
        n, k = self.X.shape
        Lambda = self.l2_reg * np.eye(k)
        # Compute the posterior mean and covariance for the linear coefficients.
        V_inv = Lambda + self.X.T.dot(self.X)
        L = spla.cholesky(V_inv, lower=True)
        L_inv = spla.solve_triangular(L, np.eye(k), lower=True)
        self.post_cov = L_inv.T.dot(L_inv)
        self.post_beta = self.post_cov.dot(self.X.T.dot(self.y))
        # Compute the posterior parameters for the noise variance.
        self.post_a = self.prior_a + n / 2.
        self.post_b = self.prior_b + 0.5 * (
            self.y.dot(self.y) -
            self.post_beta.T.dot(V_inv.dot(self.post_beta))
        )

    def sample(self, X_pred, n_samples=1, target=False):
        """Implementation of abstract base class methods."""
        X_pred = np.atleast_2d(X_pred)
        mean, cov = self.predict(X_pred)
        if target:
            cov += self.noise_level * np.eye(X_pred.shape[0])
        return multivariate_normal_sampler(mean, cov, n_samples)

    def predict(self, X_pred, diagonal=False):
        """Implementation of abstract base class method."""
        # Computes the mean and covariance according to the Bayesian linear
        # regression model of the outputs at the given inputs. This only
        # produces the covariance accounting for uncertainty in the linear
        # coefficients and does not include measurement noise uncertainty.
        # Notice that the marginal distribution of the linear coefficients is a
        # multivariate t-distribution whose covariance we can compute directly.
        #
        # References for computing this marginal covariance:
        #     https://en.wikipedia.org/wiki/Multivariate_t-distribution
        #     https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution
        a, b = self.post_a, self.post_b
        Omega = (2.*a / (2.*a - 2.)) * b / a * self.post_cov
        mean = X_pred.dot(self.post_beta)
        B = Omega.dot(X_pred.T)
        if diagonal:
            cov = np.sum(X_pred * B.T, axis=1)
        else:
            cov = X_pred.dot(B)
        return mean, cov

    def sample_parameters(self, n_samples=1):
        """Implementation of abstract base class method."""
        lam = np.random.gamma(self.post_a, 1. / self.post_b, size=(n_samples, ))
        sigma_sq = 1. / lam
        W = np.zeros((n_samples, len(self.post_beta)))
        for i in range(n_samples):
            W[i] = multivariate_normal_sampler(self.post_beta, sigma_sq[i] * self.post_cov)
        return W, sigma_sq

    def grad_input(self, x):
        """Compute the gradient of the mean function and the standard deviation
        function at the provided input.
        """
        d_V = self.post_cov.dot(x)
        V = x.dot(d_V)
        d_sd = 1. / (2. * np.sqrt(V)) * d_V
        return self.post_beta, d_sd
