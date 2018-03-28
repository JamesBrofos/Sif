import numpy as np
import scipy.linalg as spla


def sigmoid(z):
    """The sigmoid is a squashing function that compresses the entire real line
    into the unit interval.
    """
    return 1. / (1. + np.exp(-z))


class BayesianLogisticRegression:
    """Bayesian Logistic Regression Model Class

    This module implements a Bayesian logistic regression model. This class
    uses the Laplace approximation in order to draw samples from the posterior
    and characterize uncertainty. The class also uses Newton-Raphson iterations
    in order to efficiently converge to the maximum a posteriori estimate
    (which is coincidentally used as the mean of the Laplace approximation).
    """
    def __init__(self, tol=1e-5, max_iter=1000, l2_reg=0.):
        """Initialize the parameters of the Bayesian logistic regression object.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.l2_reg = l2_reg

    def fit(self, X, y):
        """Fit the parameters of the Bayesian logistic regression model."""
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

            # Check for convergence.
            if np.linalg.norm(delta) < self.tol:
                L_inv = spla.solve_triangular(L, np.eye(self.X.shape[1]), lower=True)
                self.cov = L_inv.T.dot(L_inv)
                break
            else:
                v = self.__objective
                print("Iteration: {}. Objective value: {:.4f}".format(i+1, v))

    def predict(self, X):
        """Predict class membership for the provided input. This is achieved
        via Monte Carlo integration over the link function for a specific
        univariate normal using the Laplace approximation.
        """
        n = X.shape[0]
        X = np.hstack((np.ones((n, 1)), X))
        mu = X.dot(self.beta)
        sigma_sq = np.array([x.dot(self.cov.dot(x)) for x in X])
        V = np.zeros((n, ))
        for i in range(n):
            A = np.random.normal(mu[i], sigma_sq[i], size=(10000, ))
            V[i] = sigmoid(A).mean()
        return V

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
        D = np.diag(p * (1. - p))
        return self.X.T.dot(D.dot(self.X)) + self.l2_reg * np.eye(self.beta.shape[0])


