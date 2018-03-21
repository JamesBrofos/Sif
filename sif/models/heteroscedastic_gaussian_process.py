import numpy as np
import scipy.stats as spst
import scipy.linalg as spla
from .gaussian_process import GaussianProcess
from ..samplers import multivariate_normal_sampler


class HeteroscedasticGaussianProcess:
    """Heteroscedastic Gaussian Process Class

    This class allows us to model heteroscedastic noise in regression datasets.
    This is necessary when the uniform noise level assumption of the
    homoscedastic Gaussian process does not appear to be valid. The class
    achieves heteroscedasticity by leveraging two Gaussian processes:

        1. The first Gaussian process models the expectation of the regressed
           data. This allows us to characterize the prediction of the model.
        2. The second models the noise level at any given input. I believe that
           this should output the diagonal elements of the covariance matrix
           during sampling.

    This class implements the heteroscedastic Gaussian process technique
    described in "Most Likely Heteroscedastic Gaussian Process Regression" by
    Kersting et al.
    """
    def __init__(
            self,
            mean_kernel,
            noise_kernel,
            mean_level=1e-6,
            noise_level=1e-6,
            max_iter=100,
            tol=0.01
    ):
        """Initialize the parameters of the heteroscedastic Gaussian process."""
        self.mean_kernel = mean_kernel
        self.noise_kernel = noise_kernel
        self.mean_level = mean_level
        self.noise_level = noise_level
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        # Fit the Gaussian process that models expectations.
        self.mean_gp = GaussianProcess(
            self.mean_kernel, noise_level=self.mean_level, prior_mean=y.mean()
        )
        self.mean_gp.fit(X, y)
        # Draw samples from the posterior predictive distribution. This allows
        # us to characterize the noise level.
        n_samples = 1000
        samples = self.mean_gp.sample(X, n_samples, True)
        # Iterate until convergence or the maximum number of iterations is
        # exceeded.
        prev_ll = self.mean_gp.log_likelihood
        for i in range(self.max_iter):
            # Compute the estimated residuals and model their logarithm to
            # ensure that we get positive estimates of the variance. By
            # computing the squared difference of the samples from the target
            # (divided by two) we obtain an unbiased estimator of the variance.
            # This assumes that both the samples and the targets are drawn from
            # the same underlying distribution.
            V = np.mean((samples - y) ** 2 / 2., axis=0)
            log_V = np.log(V)
            # Fit the noise Gaussian process.
            self.noise_gp = GaussianProcess(
                self.noise_kernel,
                noise_level=self.noise_level,
                prior_mean=log_V.mean()
            )
            self.noise_gp.fit(X, log_V)
            # Compute the current log-likelihood and assess whether or not it
            # has improved by more than one percent. If it has not, we generate
            # posterior samples from the predictive distribution of the
            # heteroscedastic Gaussian process.
            cur_ll = self.log_likelihood
            if np.abs((prev_ll - cur_ll) / prev_ll) < self.tol:
                break
            else:
                prev_ll = cur_ll
                samples = self.sample(X, n_samples=n_samples)
            # Show diagnostics.
            print(
                "Heteroscedastic Gaussian process iteration: {}. "
                "Log-likelihood: {:.4f}".format(
                    i, cur_ll
            ))

    def predict(self, X_pred):
        """Prediction method for the heteroscedastic Gaussian process."""
        # Compute the mean and variance of the expectation Gaussian process.
        mean, cov = self.mean_gp.predict(X_pred)
        # Compute the noise Gaussian process to achieve heteroscedasticity. We
        # assign this noise level to the diagonal elements of the expectation
        # covariance matrix.
        noise = np.exp(self.noise_gp.predict(X_pred)[0])
        for i in range(X_pred.shape[0]):
            # TODO: Is it a good idea to check if the noise variance exceeds the
            #       variance of the mean Gaussian process?
            if noise[i] > cov[i, i]:
                cov[i, i] = noise[i]
        # Ensure that we haven't broken the positive definiteness of the
        # covariance.
        eigs = np.linalg.eigvals(cov)
        if np.any(eigs < 0):
            raise ValueError("Invalid heteroscedastic covariance matrix.")
        return mean, cov

    def sample(self, X_pred, n_samples=1):
        """Sampling process for the heteroscedastic Gaussian process."""
        mean, cov = self.predict(X_pred)
        return multivariate_normal_sampler(mean, cov, n_samples)

    @property
    def log_likelihood(self):
        """Log-likelihood of the heteroscedastic Gaussian process."""
        # Compute the mean and covariance of the heteroscedastic Gaussian
        # process.
        mean, cov = self.predict(self.mean_gp.X)
        # Compute the Cholesky decomposition of the covariance matrix and the
        # centralized targets.
        L = spla.cholesky(cov, lower=True)
        y_tilde = self.mean_gp.y - mean
        # Compute the log-likelihood using numerically stable operations.
        alpha = spla.cho_solve((L, True), y_tilde)
        beta = y_tilde.dot(alpha)
        return -1. * (
            0.5 * beta +
            np.sum(np.log(np.diag(L))) +
            0.5 * self.mean_gp.X.shape[0] * np.log(2.*np.pi)
        )



