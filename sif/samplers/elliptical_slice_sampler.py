import numpy as np
import tensorflow as tf


class EllipticalSliceSampler:
    """Elliptical Slice Sampler Class"""
    def __init__(self, model, data_X, data_y):
        """Initialize the parameters of the elliptical slice sampler object."""
        self.model = model
        self.n_dim = model.kernel.n_dim
        self.data_X, self.data_y = data_X, data_y

    def __sample(self, length_scale, amplitude, noise_level, sess):
        # Choose the ellipse.
        nu_length_scale = np.random.normal(-3., 1., size=(self.n_dim, ))
        nu_amplitude = np.random.normal(0.1, 0.2)
        nu_noise_level = np.random.normal(-3., 1.)
        # Compute log-likelihood threshold.
        log_u = np.log(np.random.uniform())
        log_y = sess.run(self.model.log_likelihood, {
            self.model.model_X: self.data_X,
            self.model.model_y: self.data_y,
            self.model.kernel.length_scale: np.exp(length_scale),
            self.model.kernel.amplitude: np.exp(amplitude),
            self.model.noise_level: np.exp(noise_level)
        }) + log_u
        # Draw an initial proposal, also defining a bracket.
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        while True:
            prop_length_scale, prop_amplitude, prop_noise_level = (
                length_scale*np.cos(theta) + nu_length_scale*np.sin(theta),
                amplitude*np.cos(theta) + nu_amplitude*np.sin(theta),
                noise_level*np.cos(theta) + nu_noise_level*np.sin(theta)
            )
            log_f = sess.run(self.model.log_likelihood, {
                self.model.model_X: self.data_X,
                self.model.model_y: self.data_y,
                self.model.kernel.length_scale: np.exp(prop_length_scale),
                self.model.kernel.amplitude: np.exp(prop_amplitude),
                self.model.noise_level: np.exp(prop_noise_level)
            })
            if log_f > log_y:
                # Accept the proposal.
                return prop_length_scale, prop_amplitude, prop_noise_level
            else:
                # Shrink the bracket and try a new point.
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

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
        samples_length_scale = np.zeros((total_samples, self.n_dim))
        samples_amplitude = np.zeros((total_samples, ))
        samples_noise_level = np.zeros((total_samples, ))
        # Set the initial samples.
        samples_length_scale[0] = np.log(np.ones((self.n_dim, )) / 10.)
        samples_amplitude[0] = np.log(1.)
        samples_noise_level[0] = np.log(0.1)
        # Perform sampling with the elliptical slice sampling technique.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, total_samples):
                print("Sampling iteration: {}.".format(i))
                (
                    samples_length_scale[i],
                    samples_amplitude[i],
                    samples_noise_level[i]
                ) = self.__sample(
                    samples_length_scale[i-1],
                    samples_amplitude[i-1],
                    samples_noise_level[i-1],
                    sess
                )
        return (
            np.exp(samples_length_scale[burnin:]),
            np.exp(samples_amplitude[burnin:]),
            np.exp(samples_noise_level[burnin:])
        )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import lognorm
    from sif.kernels import SquaredExponentialKernel
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
    y = np.cos(10.*X) / (X + 1.)
    X_pred = np.atleast_2d(np.linspace(-1., 1., num=500)).T
    # Create the Gaussian process object.
    gp = GaussianProcess(SquaredExponentialKernel(1))

    # Now sample using the elliptical slice sampler.
    sampler = EllipticalSliceSampler(gp, X, y)
    length_scale, amplitude, noise_level = sampler.sample(100)
