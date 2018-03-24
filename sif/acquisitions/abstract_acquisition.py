import multiprocessing
import numpy as np
import sobol_seq
from abc import abstractmethod
from scipy.optimize import fmin_l_bfgs_b


class AbstractAcquisitionFunction:
    """Abstract Gradient-Based Acquisition Function

    Many acquisition functions utilized by Bayesian optimization are
    differentiable with respect to the input parameters. These gradients can be
    leveraged in order to find the maxima of the acquisition function. This will
    generally lead to better maxima being identified than if one were simply to
    sample randomly from the unit hypercube and compute the acquisition
    function. In order to identify the maxima of the acquisition function, this
    class utilizes the BFGS algorithm.

    Nonetheless, it is worth emphasizing that acquisition functions, even if
    they are differentiable, will still be multimodal. As a result, it is
    necessary to perform multiple random restarts from different locations
    within the unit hypercube.
    """
    def __init__(self, models):
        """Initialize the parameters of the abstract acquisition function
        object.

        Parameters:
            models (AbstractProcess): A list of Gaussian process models that
                interpolates the observed data. Each element of the list should
                correspond to a different configuration of kernel hyperparameters.
        """
        if not isinstance(models, list):
            self.models = [models]
        else:
            self.models = models
        self.n_model = len(self.models)

    def __negative_acquisition_function(self, params):
        """This function simply computes the negative of the acquisition
        function. This is required since the BFGS algorithm will seek to
        minimize the function rather than maximize it; therefore, the find the
        maxima of the acquisition function, in practice we find the minima of
        the negative of the acquisition function.

        Parameters:
            params (numpy array): An input from the unit hypercube at which the
                acquisition function will be computed.

        Returns:
            The negative of the acquisition function.
        """
        return -self.evaluate(np.atleast_2d(params))

    def __negative_acquisition_function_grad(self, params):
        """This function simply computes the negative of the gradient of the
        acquisition function.

        Parameters:
            params (numpy array): An input from the unit hypercube at which the
                gradient will be computed.

        Returns:
            The negative of the gradient of the acquisition function.
        """
        return -self.grad_input(np.atleast_2d(params))

    def maximize(self, x_cand):
        """Helper function that leverages the BFGS algorithm with a bounded
        input space in order to converge the maximum of the acquisition function
        using gradient ascent. Notice that this function returns the point in
        the original input space, not the point in the unit hypercube.

        Parameters:
            x_cand (numpy array): An array indicating the starting position for
                the BFGS optimization routine. This allows us to identify the
                highest point of the acquisition function.

        Returns:
            A tuple containing first the numpy array representing the input in
                the unit hypercube that minimizes the negative of the
                acquisition function (or, equivalently, maximizes the
                acquisition function) as well as the value of the acquisition
                function at the discovered maximum.
        """
        # Number of dimensions.
        k = self.models[0].X.shape[1]
        # Bounds on the search space used by the BFGS algorithm.
        bounds = [(0., 1.)] * k
        # Call the BFGS algorithm to perform the maximization.
        res = fmin_l_bfgs_b(
            self.__negative_acquisition_function,
            x_cand,
            fprime=self.__negative_acquisition_function_grad,
            bounds=bounds,
            disp=0
        )
        return res

    def select(self):
        """Implementation of abstract base class method."""
        # Number of dimensions.
        k = self.models[0].X.shape[1]
        # Compute the number of evaluations to perform. As a heuristic, we use
        # ten times the number of hyperparameters.
        n_evals = 10 * k

        # Create random points in the vicinity of the optimal input. This is for
        # exploitation purposes.
        x_opt = self.models[0].X[self.models[0].y.argmax()]
        X_exp = np.clip(np.random.normal(scale=1e-3, size=(50 * k, k)) + x_opt, 0., 1.)
        # Create a large grid of points on which to evaluate the acquisition
        # function. By only computing the diagonal elements, this computation is
        # relatively fast.
        X_grid = np.vstack((np.random.uniform(size=(10000 * k, k)), X_exp))
        acq_grid = self.evaluate(X_grid).ravel()
        idx_grid = np.argsort(acq_grid)[-n_evals:]
        X_cand = X_grid[idx_grid]

        # Initialize the best acquisition value to negative infinity. This will
        # allow any fit of the data to be better.
        best_acq = -np.inf
        # For the specified number of iterations, try to maximize the
        # acquisition function using random search or randomly initialized
        # gradient ascent.
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = [pool.apply_async(self.maximize, args=(X_cand[i], )) for i in range(n_evals)]
        best_res = max(results, key=lambda res: -res.get()[1][0]).get()
        best_x, best_acq = best_res[0], -best_res[1][0]
        pool.close()
        print(
            "Best acquisition function before optimization: {:.4f}. "
            "After optimization: {:.4f}.".format(acq_grid.max(), best_acq)
        )

        # Return the input that maximizes the acquisition function and the value
        # of the acquisition function at that point.
        return best_x, best_acq

    @abstractmethod
    def evaluate(self, X, integrate=True):
        """Evaluate the acquisition function at the specified inputs. Unlike the
        gradient computation for the acquisition function, this method supports
        matrix-like inputs representing multiple locations at which to evaluate
        the acquisition function.

        Parameters:
            X (numpy array): A two-dimensional numpy array that represents the
                row-wise inputs in the unit hypercube that should be used as
                input to the acquisition function.

        Returns:
            The value of the acquisition function at the specified inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def grad_input(self, x):
        """Compute the gradient of the acquisition function with respect to the
        inputs. This method returns the gradient of the acquisition, not the
        gradient of the negative of the acquisition function.

        Parameters:
            x (numpy array): An input in the unit hypercube at which the
                gradient should be computed.

        Returns:
            The gradient of the acquisition function at the specified input.
        """
        raise NotImplementedError()
