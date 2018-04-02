from abc import abstractmethod


class AbstractModel:
    """Absract Model Class

    This class implements the abstract model object. For Bayesian inference, a
    model consists of a mechanism for fitting the model, a method for
    performing (preferably non-stochastic) prediction given new inputs, and a
    method for generating samples from the Bayesian posterior over target
    variables.
    """
    @abstractmethod
    def fit(self, X, y):
        """Fit the parameters of the model based on the available training
        data. This method will be implemented by every model in order to
        achieve the learning that each algorithm aims to perform.

        Parameters:
            X (numpy array): A design matrix of features. The shape of this
                matrix should be the number of observations by the
                dimensionality of the feature representation.
            y (numpy array): A vector containing the targets of prediction. For
                regression problems, this is simply the value to be predicted.
                For classification tasks, this is a binary indicator for
                class-one membership.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X_pred, diagonal=False):
        """Leverage Bayesian posterior inference to compute the predicted mean
        and variance of a given set of inputs given the available training
        data. Notice that it is necessary to first fit the process model before
        posterior inference can be performed.

        Parameters:
            X_pred (numpy array): A design matrix of features. The shape of
                this matrix should be the number of observations by the
                dimensionality of the feature representation.
            diagonal (boolean): An indicator for whether or not the full
                covariance for the prediction should be returned. It is much
                faster to set this to True when the off-diagonal covariances
                are not required.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, X, n_samples=1, target=False):
        """Sample output variables from the predictive posterior distribution
        of the process.

        Parameters:
            X (numpy array): A design matrix of features. The shape of this
                matrix should be the number of observations by the
                dimensionality of the feature representation.
            n_samples (int): The number of posterior samples to generate.
            target (boolean): An indicator for whether or not we want to
                generate samples from the target or from a latent function
                interpolation. (There must be a better way to say this.) For
                instance, in regression settings, when true this generated
                targets, not samples of the posterior mean. In classification,
                these are binary samples, not samples of the probability of
                class one membership.
        """
        raise NotImplementedError()
