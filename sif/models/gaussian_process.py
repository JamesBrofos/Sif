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
        self.noise_level = tf.Variable(1., dtype=tf.float64)

        # Covariances.
        self.cov = (
            kernel.covariance(self.model_X, self.model_X) +
            self.noise_level * tf.eye(
                tf.shape(self.model_X)[0], dtype=tf.float64
            )
        )
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

        # We can use sampling for inference as well. Now, I was reading the
        # source code of a certain other Gaussian process library and it seems
        # that their samples are drawn from a noiseless kernel. We'll replicate
        # that approach here.
        self.L_pred = tf.cholesky(
            self.model_cov_pred + 1e-6 * tf.eye(
                tf.shape(self.cov_pred)[0], dtype=tf.float64
            )
        )
        self.dens_pred = MultivariateNormalTriL(
            loc=tf.squeeze(self.model_y_pred), scale_tril=self.L_pred
        )
        # Compute the log-likelihood of the data under the Gaussian process
        # model with the given length scales, amplitude, and noise level of the
        # kernel.
        self.log_likelihood = self.dens.log_prob(tf.squeeze(self.model_y))
        # Sample target variables from the predictive posterior distribution of
        # the Gaussian process.
        self.n_samples = tf.placeholder(tf.int32)
        self.sample = self.dens_pred.sample(self.n_samples)

