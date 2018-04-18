import numpy as np
import scipy.linalg as spla
import scipy.sparse as spsp
from .generalized_linear_model import GeneralizedLinearModel
from ..samplers import multivariate_normal_sampler


def sigmoid(z):
    """The sigmoid is a squashing function that compresses the entire real line
    into the unit interval.
    """
    return 1. / (1. + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function with respect to its input."""
    p = sigmoid(z)
    return p * (1. - p)

class BayesianLogisticRegression(GeneralizedLinearModel):
    """Bayesian Logistic Regression Model Class

    This module implements a Bayesian logistic regression model. This class
    uses the Laplace approximation in order to draw samples from the posterior
    and characterize uncertainty. The class also uses Newton-Raphson iterations
    in order to efficiently converge to the maximum a posteriori estimate
    (which is coincidentally used as the mean of the Laplace approximation).
    """
    def __init__(self, l2_reg=0., tol=1e-5, max_iter=100):
        """Initialize the parameters of the Bayesian logistic regression object.
        """
        super().__init__(l2_reg)
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """Implementation of abstract base class method."""
        # Initialize vector of linear coefficients. Notice that we add a bias
        # term ourselves.
        self.X, self.y = np.hstack((np.ones((X.shape[0], 1)), X)), y
        self.beta = np.zeros((self.X.shape[1], ))
        # Iterate until convergence.
        for i in range(self.max_iter):
            # Compute the Hessian and gradient of the objective function.
            H = self.__hessian
            g = self.__gradient
            # Use Cholesky decomposition to efficiently solve the linear system.
            L = spla.cholesky(H, lower=True)
            delta = spla.cho_solve((L, True), g)
            # Update the linear coefficients.
            self.beta += delta
            if np.any(np.isnan(self.beta)):
                raise ValueError("NaNs encountered in linear coefficients.")

            # Print diagnostics.
            v = self.__objective
            print("Iteration: {}. Objective value: {:.4f}".format(i+1, v))
            # Check for convergence.
            if np.linalg.norm(delta) < self.tol:
                break

        # Create covariance matrix for the linear coefficients under a Laplace
        # approximation.
        L_inv = spla.solve_triangular(L, np.eye(self.X.shape[1]), lower=True)
        self.cov = L_inv.T.dot(L_inv)

    def sample(self, X_pred, n_samples=1, target=False):
        """Implementation of abstract base class method."""
        X_pred = np.atleast_2d(X_pred)
        rvs = multivariate_normal_sampler(X_pred.dot(self.beta), self.cov)
        p = sigmoid(rvs)
        if target:
            return (np.random.uniform(size=p.shape) < p).astype(float)
        else:
            return p

    def predict(self, X_pred, diagonal=False):
        """Implementation of abstract base class method."""
        # Note that we use the Delta Method to obtain the variance.
        #
        # TODO: Check that this produces sane results.
        X_pred = np.atleast_2d(X_pred)
        z = X_pred.dot(self.beta)
        mean = sigmoid(z)
        D = sigmoid_derivative(z) * X_pred.T
        B = self.cov.dot(D)
        if diagonal:
            cov = np.sum(D * B.T, axis=0)
        else:
            cov = D.T.dot(B)
        return mean, cov

    def sample_parameters(n_samples=1):
        """Implementation of abstract base class method."""
        return multivariate_normal_sampler(self.beta, self.cov, n_samples)

    @property
    def __objective(self):
        p = np.clip(sigmoid(self.X.dot(self.beta)), 1e-7, 1.-1e-7)
        ll = np.sum(self.y * np.log(p) + (1. - self.y) * np.log(1. - p))
        R = -0.5 * self.l2_reg * np.sum(self.beta ** 2)
        return ll + R

    @property
    def __gradient(self):
        p = sigmoid(self.X.dot(self.beta))
        return self.X.T.dot(self.y - p) - self.l2_reg * self.beta

    @property
    def __hessian(self):
        p = sigmoid(self.X.dot(self.beta))
        D = spsp.diags(p * (1. - p))
        return self.X.T.dot(D.dot(self.X)) + self.l2_reg * np.eye(self.beta.shape[0])


