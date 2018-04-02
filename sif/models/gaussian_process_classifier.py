import numpy as np
import scipy.linalg as spla
from .abstract_model import AbstractModel
from .bayesian_logistic_regression import sigmoid
from ..samplers import multivariate_normal_sampler


class GaussianProcessClassifier(AbstractModel):
    """Gaussian Process Classifier Class"""
    def __init__(self, kernel, tol=1e-10):
        """Initialize the parameters of the Gaussian process classifier object.
        """
        self.kernel = kernel
        self.tol = tol

    def fit(self, X, y):
        """Fit the parameters of the Gaussian process classifier based on the
        available training data. Note that this class supports binary
        classification tasks exclusively.
        """
        # Store the training data (both the inputs and the targets). We need a
        # copy of the binary target so that we avoid directly modifying that
        # variable.
        self.X, self.y = X, y.copy().ravel()
        self.y[self.y == 0.] = -1.
        t = (self.y + 1.) / 2.
        n = self.X.shape[0]
        # Compute the covariance matrix of the observed inputs.
        I = np.eye(n)
        self.K = self.kernel.cov(self.X, self.X) + 1e-6 * I
        # Initialize the posterior mode to a vector of zeros.
        f = np.zeros((n, ))
        n_iters = 0

        while True:
            W = -np.diag(self.__hess_log_prob_y_given_f(self.y, f))
            sqrt_W = np.sqrt(W)
            L = spla.cholesky(I + sqrt_W.dot(self.K.dot(sqrt_W)), lower=True)
            b = W.dot(f) + self.__grad_log_prob_y_given_f(self.y, f)
            Q = spla.cho_solve((L, True), sqrt_W.dot(self.K.dot(b)))
            a = b - sqrt_W.dot(Q)
            f = self.K.dot(a)
            obj = -0.5 * a.dot(f) + self.__log_prob_y_given_f(self.y, f)
            if n_iters == 0:
                prev_obj = obj
            else:
                if np.abs(prev_obj - obj) < self.tol:
                    self.W, self.L, self.f, self.a = W, L, f, a
                    break
                else:
                    prev_obj = obj
            n_iters += 1

    def predict(self, X_pred, diagonal=False):
        """Implementation of abstract base class method."""
        # We generate a large number of samples in order to obtain low variance
        # estimates of the mean and variance.
        S = self.sample(X_pred, 10000)
        mean = S.mean(axis=0)
        if diagonal:
            cov = S.var(axis=0)
        else:
            cov = np.cov(S.T)
        return mean, cov

    def sample(self, X_pred, n_samples=1, target=False):
        """Implementation of abstract base class method."""
        # Compute the cross covariance between training and the requested
        # inference locations. Also compute the covariance matrix of the
        # observed inputs and the covariance at the inference locations.
        K_pred = self.kernel.cov(X_pred, X_pred)
        K_cross = self.kernel.cov(X_pred, self.X)
        v = spla.solve_triangular(self.L, np.sqrt(self.W).dot(K_cross.T), lower=True)
        # Compute the mean and covariance of the Laplace approximation.
        norm_mean = K_cross.dot(self.__grad_log_prob_y_given_f(self.y, self.f))
        norm_cov = K_pred - v.T.dot(v) + 1e-6 * np.eye(K_pred.shape[0])
        rvs = multivariate_normal_sampler(norm_mean, norm_cov, n_samples)
        p = sigmoid(rvs)
        if target:
            return (np.random.uniform(size=p.shape) < p).astype(float)
        else:
            return p

    @property
    def log_likelihood(self):
        """For Gaussian process classifiers, we compute the approximate marginal
        likelihood using a Laplace approximation.
        """
        return (
            -0.5 * self.a.dot(self.f) +
            self.__log_prob_y_given_f(self.y, self.f) -
            np.sum(np.log(np.diag(self.L)))
        )

    def __log_prob_y_given_f(self, y, f):
        return -np.log(1. + np.exp(-y*f)).sum()

    def __grad_log_prob_y_given_f(self, y, f):
        t = (y + 1.) / 2.
        pi = sigmoid(f)
        return t - pi

    def __hess_log_prob_y_given_f(self, y, f):
        pi = sigmoid(f)
        return -pi * (1. - pi)

