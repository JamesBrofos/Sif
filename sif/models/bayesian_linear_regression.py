import numpy as np
import scipy.linalg as spla
from ..samplers import multivariate_normal_sampler


class BayesianLinearRegression:
    """Bayesian Linear Regression Class"""
    def __init__(self, prior_w, prior_cov, prior_alpha=1., prior_beta=1.):
        """Initialize the parameters of the Bayesian linear regression object.
        """
        self.prior_w = prior_w
        self.prior_prec = spla.inv(prior_cov)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def fit(self, X, y):
        """Fit the parameters of the process based on the available training
        data.
        """
        # Store the training data (both the inputs and the targets).
        self.X, self.y = X, y.ravel()
        n, k = self.X.shape
        self.post_alpha = self.prior_alpha + n / 2.
        V_inv = self.prior_prec + self.X.T.dot(self.X)
        L = spla.cholesky(V_inv, lower=True)
        L_inv = spla.solve_triangular(L, np.eye(k), lower=True)
        self.post_V = L_inv.T.dot(L_inv)
        self.post_w = self.post_V.dot(
            self.prior_prec.dot(self.prior_w) + self.X.T.dot(self.y)
        )
        self.post_beta = self.prior_beta + 0.5 * (
            self.prior_w.T.dot(self.prior_prec.dot(self.prior_w)) +
            self.y.dot(self.y) -
            self.post_w.T.dot(V_inv.dot(self.post_w))
        )

    def predict(self, X_pred):
        """Computes the mean and covariance according to the Bayesian linear
        regression model of the outputs at the given inputs. This only produces
        the covariance accounting for uncertainty in the linear coefficients and
        does not include measurement noise uncertainty. Notice that the marginal
        distribution of the linear coefficients is a multivariate t-distribution
        whose covariance we can compute directly.
        """
        # References for computing this marginal covariance:
        #     https://en.wikipedia.org/wiki/Multivariate_t-distribution
        #     https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution
        a, b = self.post_alpha, self.post_beta
        Omega = (2.*a / (2.*a - 2.)) * b / a * self.post_V
        mean = X_pred.dot(self.post_w)
        cov = X_pred.dot(Omega.dot(X_pred.T))
        return mean, cov

    def sample(self, n_samples=1):
        """Samples the linear coefficients and noise variance given the observed
        data.
        """
        lam = np.random.gamma(
            self.post_alpha, 1. / self.post_beta, size=(n_samples, )
        )
        sigma_sq = 1. / lam
        W = np.zeros((n_samples, len(self.prior_w)))
        for i in range(n_samples):
            W[i] = multivariate_normal_sampler(
                self.post_w, sigma_sq[i] * self.post_V
            )
        return W, sigma_sq
