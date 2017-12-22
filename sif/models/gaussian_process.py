import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalTriL

class GaussianProcess:
    """Gaussian Process Class"""
    def __init__(self, kernel):
        """Initialize the parameters of the Gaussian process object."""
        # Kernels for inference.
        self.kernel = kernel
        n_dim = self.kernel.n_dim
        # Create placeholders for the training and prediction variables of the
        # Gaussian process.
        self.model_X = tf.placeholder(tf.float64, shape=[None, n_dim])
        self.model_y = tf.placeholder(tf.float64, shape=[None, 1])
        self.model_X_pred = tf.placeholder(tf.float64, shape=[None, n_dim])
        # Noise level of the Gaussian process.
        self.noise_level = tf.Variable(1e-6, dtype=tf.float64)

        # Covariances.
        self.cov = kernel.covariance(self.model_X, self.model_X)
        self.cov += self.noise_level * tf.eye(tf.shape(self.cov)[0], dtype=tf.float64)
        self.cov_cross = kernel.covariance(self.model_X_pred, self.model_X)
        self.cov_pred = kernel.covariance(self.model_X_pred, self.model_X_pred)
        # Create training variables for the Gaussian process.
        self.L = tf.cholesky(self.cov)
        self.dens = MultivariateNormalTriL(scale_tril=self.L)
        self.alpha = tf.cholesky_solve(self.L, self.model_y)

        # Posterior expectation and variance.
        self.model_y_pred = tf.matmul(self.cov_cross, self.alpha)
        v = tf.matrix_triangular_solve(self.L, tf.transpose(self.cov_cross))
        self.model_cov_pred = self.cov_pred - tf.matmul(v, v, transpose_a=True)

        # We can use sampling for inference as well.
        noise_pred = self.noise_level * tf.eye(tf.shape(self.cov_pred)[0], dtype=tf.float64)
        self.L_pred = tf.cholesky(self.model_cov_pred + noise_pred)
        self.dens_pred = MultivariateNormalTriL(
            loc=tf.squeeze(self.model_y_pred), scale_tril=self.L_pred
        )
        self.log_likelihood = self.dens.log_prob(tf.squeeze(self.model_y))
        self.n_samples = tf.placeholder(tf.int32)
        self.sample = self.dens_pred.sample(self.n_samples)

    # def sample(self, n_samples):
    #     """Sample target variables from the predictive posterior distribution of
    #     the Gaussian process.
    #     """
    #     return self.dens_pred.sample(n_samples)

    # @property
    # def log_likelihood(self):
    #     """Compute the log-likelihood of the data under the Gaussian process
    #     model with the given length scales, amplitude, and noise level of the
    #     kernel.
    #     """
    #     return self.dens.log_prob(tf.squeeze(self.model_y))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sif.kernels import SquaredExponentialKernel

    # Create random data.
    X = np.random.uniform(size=(15, 1))
    y = np.cos(10.*X) / (X + 1.)
    X_pred = np.atleast_2d(np.linspace(-0.25, 1., num=500)).T
    # Create the Gaussian process object.
    gp = GaussianProcess(SquaredExponentialKernel(1))

    # Now TensorFlow!
    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(tf.global_variables_initializer())
        # Compute the posterior mean and variance of the Gaussian process.
        y_pred, cov_pred = sess.run(
            [gp.model_y_pred, gp.model_cov_pred], {
                gp.model_X: X,
                gp.model_y: y,
                gp.model_X_pred: X_pred
            }
        )
        std_pred = np.sqrt(np.diag(cov_pred))
        # Compute the log-likelihood of the Gaussian process.
        ll = sess.run(
            gp.log_likelihood, {
                gp.model_X: X,
                gp.model_y: y
            }
        )
        print("Gaussian process log-likelihood: {:.4f}".format(ll))
        # Sample from the Gaussian process posterior.
        n_samples = 100
        samples = sess.run(gp.sample, {
            gp.model_X: X,
            gp.model_y: y,
            gp.model_X_pred: X_pred,
            gp.n_samples: n_samples
        })

    # Visualize if desired.
    if True:
        plt.figure()
        plt.plot(X_pred.ravel(), y_pred.ravel(), "b-")
        plt.plot(X_pred.ravel(), y_pred.ravel() + 2. * std_pred, "b--")
        plt.plot(X_pred.ravel(), y_pred.ravel() - 2. * std_pred, "b--")
        for i in range(n_samples):
            plt.plot(X_pred.ravel(), samples[i], "b-", alpha=0.05)
        plt.plot(X.ravel(), y.ravel(), "r.")
        plt.grid()
        plt.show()

