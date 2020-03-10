import numpy as np
from abc import ABC, abstractmethod
from numpy.random import RandomState
from scipy.optimize import minimize

from util.defs import *
from util.util import random_x_sample
from acquisition.acquisition_functions import AcquisitionFunction


class AcquisitionOptimizer(ABC):
    """
    Optimizer which uses an acquisition funtion to find optimal inputs
    to a surrogate function. These inputs should then be used to sample
    some true objective function.
    """
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction):
        self.surrogate = surrogate
        self.acquisition = acquisition

    @abstractmethod
    def optimize(self, X: NDArray, bounds: NDArray, param_types: Sequence[ParamType]) -> NDArray:
        """ Optimize the acquisition function with respect to 
        inputs to the the surrogate.
        
        Args:
            X (NDArray): All samples evaluated on the loss so far.
            bounds (NDArray): A dimension (P, 2) Matrix where each row is for a 
                hyperparameter and the columns are the min/max bounds of 
                each hyperparameter, respectively.
            param_types (Sequence): A sequence of types for each hyperparameter. 
                Order should match the rows of `bounds`.
        
        Returns:
            NDArray: 1d array which is the optimal next value of `X` to 
                try on the objective. Returned array will be of floats,
                but hyperparameter indices specified as integers will be 
                rounded to the nearest integer (e.g. 5.49 to 5.0).
        """
        pass

    def _enforce_param_types(self, x: NDArray, param_types: Sequence[ParamType]) -> NDArray:
        assert x.ndim == 1, f"Assuming x to be a 1-d vector, not {x.shape}"
        for col in range(len(x)):
            if param_types[col] == ParamType.Disc:
                x[col] = np.round(x[col], decimals=0)


class RandomAcquisitionOpt(AcquisitionOptimizer):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction, rand: Optional[RandomState], sample_size=10):
        """ 
        Args:
            surrogate (object): The surrogate model which will be sampled.
            acquisition (AcquisitionFunction): The acquisition function which will
                be used in order to evaluate samples on the surrogate.
            rand (RandomObject): Random seed (defaults to `numpy.random`)
            sample_size (int): Number of random samples to use for sampling the
                acquisition function.
        """
        super(RandomAcquisitionOpt, self).__init__(surrogate, acquisition)
        self.rand = rand if rand else np.random
        self.sample_size = sample_size


    def optimize(self, X: NDArray, bounds: NDArray, param_types: Sequence[ParamType]) -> NDArray:
        assert len(bounds) == len(param_types), "Must provide a ParamType and bound for each parameter."

        # Find best surrogate score so far
        yhat = self.surrogate.predict(X)
        best_y = yhat.min()

        X_samp = random_x_sample(bounds, param_types, self.sample_size, self.rand)

        # find acquisition function value for each sample
        scores: NDArray = self.acquisition.acquire(X_samp, self.surrogate, best_y)
        
        # returns 1d vector (handle reshaping elsewhere)
        best_x = X_samp[np.argmin(scores)]
        self._enforce_param_types(best_x, param_types)
        return best_x


class ConstrainedAcquisitionOpt(AcquisitionOptimizer):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction, rand: Optional[RandomState], n_restarts = 10):
        """ 
        Args:
            surrogate (object): The surrogate model which will be sampled.
            acquisition (AcquisitionFunction): The acquisition function which will
                be used in order to evaluate samples on the surrogate.
            rand (RandomObject): Random seed (defaults to `numpy.random`)
            n_restarts (int): Number of points at which to restart the optimization. Higher
                values will mean longer rounds, but may lead to finding better results.
        """
        super(ConstrainedAcquisitionOpt, self).__init__(surrogate, acquisition)
        self.rand = rand if rand else np.random
        self.n_restarts = n_restarts

# Constrained optimization function inspired by: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
    def optimize(self, X: NDArray, bounds: NDArray, param_types: Sequence[ParamType]) -> NDArray:
        # Find best surrogate prediction so far
        y_hat = self.surrogate.predict(X)
        best_y = y_hat.min()

        n_params = X.shape[1]

        # Want to do some random restarts with our optimization.
        # Pick X to be random parameter values within their respective bounds
        starting_points = self.rand.uniform(bounds[:, 0], bounds[:, 1], size=(self.n_restarts, n_params))
        best_acq: Optional[float] = None
        best_x: Optional[NDArray] = None

        for start_point in starting_points:
            start_point = start_point

            opt = minimize(self.acquisition.acquire, 
                           x0=start_point, 
                           args=(self.surrogate, best_y), 
                           method="L-BFGS-B",
                           bounds=bounds)
            acq_new = opt.fun
            x_new = opt.x

            if best_acq is None:
                best_acq = acq_new
                best_x = x_new
            elif acq_new < best_acq:
                best_acq = acq_new 
                best_x = x_new 
        
        # returns 1d vector (handle reshaping elsewhere)
        self._enforce_param_types(best_x, param_types)
        return best_x
