import numpy as np
from scipy.stats import norm
from .improvement_acquisition import ImprovementAcquisitionFunction


class ImprovementProbability(ImprovementAcquisitionFunction):
    """Improvement Probability Acquisition Function Class

    The improvement probability acquisition function leverages the idea of
    probability of improvement to select the next hyperparameter configuration
    to evaluate. This acquisition function accumulates the probability mass
    above the current best observation given the posterior distribution computed
    by the surrogate model. Because it does not weight the extent of the
    improvement (only its probability), the probability of improvement can be
    prone to exploiting too much in practical applications.
    """
    def evaluate(self, X, integrate=True):
        """Implementation of abstract base class method."""
        pis = norm.cdf(self.score(X)[0])
        if integrate:
            return pis.mean(axis=0)
        else:
            return pis

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        m, k = self.n_models, x.shape[1]
        gammas, means, sds = self.score(x)
        grads = np.zeros((m, k))
        for i, mod in enumerate(self.models):
            d_mean, d_sd = mod.grad_input(x)
            d_gamma = (d_mean - gammas[i] * d_sd) / sds[i]
            grads[i] = norm.pdf(gammas[i]) * d_gamma
        return grads.mean(axis=0)


