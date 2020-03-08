import numpy as np
from abc import ABC, abstractmethod
from numpy.random import RandomState
from scipy.optimize import minimize

from util.defs import *
from acquisition.acquisition_functions import AcquisitionFunction


class AcquisitionOptimizer(ABC):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction):
        self.surrogate = surrogate
        self.acquisition = acquisition

    @abstractmethod
    def optimize(self, X: NDArray, bounds: NDArray, param_types: Sequence[ParamType]) -> NDArray:
        pass

    def enforce_param_types(self, x: NDArray, param_types: Sequence[ParamType]) -> NDArray:
        assert x.ndim == 1, f"Assuming x to be a 1-d vector, not {x.shape}"
        for col in range(len(x)):
            if param_types[col] == ParamType.Disc:
                x[col] = np.round(x[col], decimals=0)


class RandomAcquisitionOpt(AcquisitionOptimizer):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction, rand: Optional[RandomState], sample_size=10):
        super(RandomAcquisitionOpt, self).__init__(surrogate, acquisition)
        self.rand = rand if rand else np.random
        self.sample_size = sample_size


    def optimize(self, X: NDArray, bounds: NDArray, param_types: Sequence[ParamType]) -> NDArray:
        assert len(bounds) == len(param_types), "Must provide a ParamType and bound for each parameter."

        # Find best surrogate score so far
        yhat = self.surrogate.predict(X)
        best_y = yhat.min()
        n_params = X.shape[1] 
        X_samp = []

        # random search of the domain
        for _ in range(self.sample_size):
            x_i = np.array([self.rand_sample_in_bounds(bounds[j], param_types[j]) for j in range(n_params)])
            X_samp.append(x_i)
        
        X_samp: NDArray = np.array(X_samp)

        # find acquisition function value for each sample
        scores: NDArray = self.acquisition.acquire(X_samp, self.surrogate, best_y)
        
        # returns 1d vector (handle reshaping elsewhere)
        best_x = X_samp[np.argmin(scores)]
        self.enforce_param_types(best_x, param_types)
        return best_x


    def rand_sample_in_bounds(self, bounds: Tuple, param_type: ParamType) -> Union[float, int]:
        # Uniform sample
        if param_type == ParamType.Cont:
            x_j = self.rand.uniform(bounds[0], bounds[1])
        elif param_type == ParamType.Disc:
            x_j = self.rand.randint(bounds[0], bounds[1])
        return x_j


class ConstrainedAcquisitionOpt(AcquisitionOptimizer):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction, rand: Optional[RandomState], n_restarts = 10):
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
            ei_new = opt.fun
            x_new = opt.x

            if best_acq is None:
                best_acq = ei_new
                best_x = x_new
            elif ei_new < best_acq:
                best_acq = ei_new 
                best_x = x_new 
        
        # returns 1d vector (handle reshaping elsewhere)
        self.enforce_param_types(best_x, param_types)
        return best_x