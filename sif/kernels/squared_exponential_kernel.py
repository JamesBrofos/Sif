import tensorflow as tf
from .abstract_kernel import AbstractKernel


class SquaredExponentialKernel(AbstractKernel):
    """Squared Exponential Kernel Class"""
    def covariance(self, model_X, model_Y):
        # Compute the squared Euclidean distance between points.
        exp_X = tf.expand_dims(model_X, 1) / self.length_scale
        exp_Y = tf.expand_dims(model_Y, 0) / self.length_scale
        dist_sq = tf.reduce_sum(tf.squared_difference(exp_X, exp_Y), 2)
        return self.amplitude * tf.exp(-dist_sq)

