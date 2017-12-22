import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist


class SquaredExponentialKernel:
    """Squared Exponential Kernel Class"""
    def __init__(self, n_dim):
        """Initialize the parameters of the squared exponential kernel object.
        """
        # Define a variable for the length scales.
        self.n_dim = n_dim
        self.amplitude = tf.Variable(1., tf.float64, dtype=tf.float64)
        self.length_scale = tf.Variable(tf.ones((self.n_dim, ), tf.float64)) / 10.

    def covariance(self, model_X, model_Y):
        # Compute the squared Euclidean distance between points.
        exp_X = tf.expand_dims(model_X, 1) / self.length_scale
        exp_Y = tf.expand_dims(model_Y, 0) / self.length_scale
        dist_sq = tf.reduce_sum(tf.squared_difference(exp_X, exp_Y), 2)
        return self.amplitude * tf.exp(-dist_sq)

if __name__ == "__main__":
    # Create random numpy matrices.
    n_dim = 5
    X = np.random.normal(size=(10, n_dim))
    Y = np.random.normal(size=(10, n_dim))
    # Create the squared exponential kernel.
    kernel = SquaredExponentialKernel(n_dim, tf.float64)
    # Compare Scipy and TensorFlow.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed = {kernel.model_X: X, kernel.model_Y: Y}
        kernel_dist_sq = sess.run(kernel.dist_sq, feed)
        scipy_dist_sq = cdist(X, Y, "sqeuclidean")
        norm = np.linalg.norm(kernel_dist_sq - scipy_dist_sq)
        print("Difference between distance matrices: {}".format(norm))
