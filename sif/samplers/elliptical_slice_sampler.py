import numpy as np
from numpy.random import multivariate_normal, uniform


class EllipticalSliceSampler:
    """Elliptical Slice Sampler Class"""
    def __init__(self, mean, covariance, log_likelihood_func):
        """Initialize the parameters of the elliptical slice sampler object."""
        self.mean, self.covariance = mean, covariance
        self.log_likelihood_func = log_likelihood_func

    def __sample(self, f):
        # Choose the ellipse.
        nu = multivariate_normal(np.zeros(self.mean.shape), cov=self.covariance)
        # Compute log-likelihood threshold.
        log_u = np.log(uniform())
        log_y = self.log_likelihood_func(f) + log_u
        # Draw an initial proposal, also defining a bracket.
        theta = uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        while True:
            fp = (f - self.mean)*np.cos(theta) + nu*np.sin(theta) + self.mean
            log_fp = self.log_likelihood_func(fp)
            if log_fp > log_y:
                # Accept the proposal.
                return fp
            else:
                # Shrink the bracket and try a new point.
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = uniform(theta_min, theta_max)

    def sample(self, n_samples, burnin=100):
        """Draws the specified number of samples using elliptical slice
        sampling. This function also allows the user to specify the number of
        burn-in iterations to perform to achieve correct sampling.
        """
        # Create vectors to store the samples of the model hyperparameters.
        total_samples = burnin + n_samples
        samples = np.zeros((total_samples, self.covariance.shape[0]))
        samples[0] = multivariate_normal(mean=self.mean, cov=self.covariance)
        # Perform sampling with the elliptical slice sampling technique.
        for i in range(1, total_samples):
            samples[i] = self.__sample(samples[i-1])
        return samples[burnin:]

