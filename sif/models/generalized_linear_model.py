import scipy.linalg as spla
from abc import abstractmethod
from .abstract_model import AbstractModel


class GeneralizedLinearModel(AbstractModel):
    """Generalized Linear Model Class"""
    def __init__(self, l2_reg):
        """Initialize the parameters of the generalized linear model object."""
        super().__init__()
        self.l2_reg = l2_reg

    @abstractmethod
    def sample_parameters(self, n_samples=1):
        """Samples the linear coefficients and other latent parameters that
        define the Bayesian generalized linear model.
        """
        raise NotImplementedError()


