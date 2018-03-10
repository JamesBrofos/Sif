import numpy as np
from scipy.stats import norm
from .improvement_acquisition import ImprovementAcquisitionFunction


class ExpectedImprovement(ImprovementAcquisitionFunction):
    """Expected Improvement Acquisition Function Class

    The expected improvement acquisition function, like the probability of
    improvement, similarly relies on the concept of improvement: That is, the
    extent to which the metric of interest can be expected under the posterior
    to exceed the current observed maximum. The expected improvement does not
    merely accumulate the probability density above the best value of the
    metric, however; instead, it weights each hyperparameter configuration
    according to the extent of the improvement. Therefore, improvement must not
    only be likely, but it must also be substantive.

    In practice, the expected improvement has been shown to out perform the
    upper confidence bound and probability of improvement acquisition functions.
    It is less likely to exploit local maxima than the probability of
    improvement is, and does not rely on its own hyperparameters unlike the
    upper confidence bound.
    """
    def evaluate(self, X, integrate=True):
        """Implementation of abstract base class method."""
        m, n = self.n_model, X.shape[0]
        gammas, means, sds = self.score(X)
        eis = np.zeros((m, n))
        for i in range(m):
            eis[i] = (
                (means[i] - self.y_best) * norm.cdf(gammas[i]) +
                sds[i] * norm.pdf(gammas[i])
            )
        if integrate:
            return eis.mean(axis=0)
        else:
            return eis

    def grad_input(self, x):
        """Implementation of abstract base class method."""
        m, k = self.n_model, x.shape[1]
        gammas, means, sds = self.score(x)
        grads = np.zeros((m, k))
        for i, mod in enumerate(self.models):
            d_mean, d_sd = mod.grad_input(x)
            d_gamma = (d_mean - gammas[i] * d_sd) / sds[i]
            grad = (
                gammas[i] * norm.cdf(gammas[i]) + norm.pdf(gammas[i])
            ) * d_sd
            grad += sds[i] * norm.cdf(gammas[i]) * d_gamma
            grads[i] = grad
        return grads.mean(axis=0)
