import numpy as np
import tensorflow as tf
from numpy.random import multivariate_normal, uniform


class EllipticalSliceSampler:
    """Elliptical Slice Sampler Class"""
    def __init__(self, mean, covariance, log_likelihood_func):
        """Initialize the parameters of the elliptical slice sampler object."""
        self.mean, self.covariance = mean, covariance
        self.log_likelihood_func = log_likelihood_func

    def __sample(self, f, sess):
        # Choose the ellipse.
        nu = multivariate_normal(np.zeros(self.mean.shape), cov=self.covariance)
        # Compute log-likelihood threshold.
        log_u = np.log(uniform())
        log_y = self.log_likelihood_func(f, sess) + log_u
        # Draw an initial proposal, also defining a bracket.
        theta = uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        while True:
            fp = (f - self.mean)*np.cos(theta) + nu*np.sin(theta) + self.mean
            log_fp = self.log_likelihood_func(fp, sess)
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
        # So let's just say that the length scales and noise level will be
        # sampled from logN(-3., 1.) and the amplitude will be sampled from a
        # logN(1/10, 1/5).
        #
        # Create vectors to store the samples of the model hyperparameters.
        total_samples = burnin + n_samples
        samples = np.zeros((total_samples, self.covariance.shape[0]))
        samples[0] = multivariate_normal(mean=self.mean, cov=self.covariance)
        # Perform sampling with the elliptical slice sampling technique.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, total_samples):
                samples[i] = self.__sample(samples[i-1], sess)
        return samples[burnin:]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import lognorm
    from sif.kernels import SquaredExponentialKernel, MaternKernel
    from sif.models import GaussianProcess

    # Visualize what a log-normal distribution looks like.
    if False:
        r = np.linspace(0., 2., num=100)
        mu, sigma = 0.1, 0.2
        plt.figure()
        plt.plot(r, lognorm.pdf(r, sigma, scale=np.exp(mu)))
        plt.grid()
        plt.show()

    # Create random data.
    X = np.random.uniform(size=(15, 1))
    y = np.random.normal(np.cos(10.*X) / (X + 1.), 0.1)
    X_pred = np.atleast_2d(np.linspace(-0.25, 1., num=500)).T
    # Create the Gaussian process object.
    gp = GaussianProcess(SquaredExponentialKernel(1))

    # Create the log-likelihood function.
    def log_likelihood_func(f, sess):
        log_length_scale, log_amplitude, log_noise_level = f[:1], f[-2], f[-1]
        return sess.run(gp.log_likelihood, {
            gp.model_X: X,
            gp.model_y: y,
            gp.kernel.length_scale: np.exp(log_length_scale),
            gp.kernel.amplitude: np.exp(log_amplitude),
            gp.noise_level: np.exp(log_noise_level)
        })

    # Now sample using the elliptical slice sampler.
    n_samples = 1000
    mean = np.array([-3., 0.1, -3.])
    covariance = np.diag(np.array([1., 0.2, 1.]))
    sampler = EllipticalSliceSampler(mean, covariance, log_likelihood_func)
    samples = np.exp(sampler.sample(n_samples))

    # Okay now sample from the Gaussian process with varying hyperparameters.
    func_samples = np.zeros((n_samples, X_pred.shape[0]))
    with tf.Session() as sess:
        for i in range(n_samples):
            func_samples[i] = sess.run(
                gp.model_y_pred, {
                    gp.model_X: X,
                    gp.model_y: y,
                    gp.model_X_pred: X_pred,
                    gp.n_samples: 1,
                    gp.kernel.length_scale: samples[i, :1],
                    gp.kernel.amplitude: samples[i, -2],
                    gp.noise_level: samples[i, -1]
                }).ravel()

    # Visualize if desired.
    if True:
        plt.figure()
        for i in range(n_samples):
            plt.plot(X_pred.ravel(), func_samples[i], "b-", alpha=5. / n_samples)
        plt.plot(X.ravel(), y.ravel(), "r.")
        plt.grid()
        plt.show()

    if False:
        plt.figure()
        plt.hist(noise_level, bins=10, normed=True)
        plt.grid()
        plt.show()

