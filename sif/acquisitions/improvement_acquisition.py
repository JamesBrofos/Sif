import numpy as np
from .abstract_acquisition import AbstractAcquisitionFunction


class ImprovementAcquisitionFunction(AbstractAcquisitionFunction):
    """Improvement-Based Acquisition Function Class

    Common acquisition function can be interpreted as improvements over the best
    seen observation so far. In particular, Thor implements the probability of
    improvement and expected improvement acquisition function, which measures
    the probability that an input will result in an improvement in the latent
    objective function and the extent to which an input can be expected to
    result in an improvement, respectively.

    As it turns out, these improvement-based acquisition functions all rely on
    retaining both the best seen value of the latent objective and on computing
    a z-score quantity that normalizes the predicted mean of the surrogate
    probabilistic model with respect to both the maximum observed value of the
    metric and the extent of uncertainty about that prediction.

    Parameters:
        models (AbstractProcess): A list of Gaussian process models that
            interpolates the observed data. Each element of the list should
            correspond to a different configuration of kernel hyperparameters.
        y_best (float): The best seen value of the metric observed so far. This
            is an optional parameter, and if it is not specified by the user
            then it will be computed directly from Thor's database (in
            particular, taking the maximum of all values of the metric recorded
            for an experiment).
    """
    def __init__(self, models, y_best=None):
        """Initialize parameters of the improvement-based acquisition function.
        """
        super().__init__(models)
        if y_best is not None:
            self.y_best = y_best
        else:
            self.y_best = self.models[0].y.max()

    def score(self, X):
        """Compute a z-score quantity for the prediction at a given input. This
        allows us to balance improvement over the current best while controlling
        for uncertainty.

        Parameters:
            X (numpy array): A numpy array representing the points at which we
                need to compute the value of the improvement-based acquisition
                function. For each row in the input, we will compute the
                associated mean and standard deviation of the mean. This latter
                quantity, alongside the best value of the metric, are then used
                to standardize the mean.

        Returns:
            A tuple containing the z-score, the mean of the metric at each
                input, and the standard deviation of the mean at each input.
        """
        m, n = self.n_models, X.shape[0]
        means, sds, gammas = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
        for i, mod in enumerate(self.models):
            # Compute the mean and standard deviation of the model's interpolant
            # of the objective function.
            means[i], var = mod.predict(X, diagonal=True)
            sds[i] = np.sqrt(var)
            # Compute z-score-like quantity capturing the excess of the mean
            # over the current best, adjusted for uncertainty in the measurement.
            gammas[i] = (means[i] - self.y_best) / sds[i]

        return gammas, means, sds
