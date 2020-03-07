
import numpy as np
from abc import ABC, abstractmethod
from numpy.random import RandomState

from util.defs import *
from acquisition.acquisition_functions import AcquisitionFunction


class AcquisitionOptimizer(ABC):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction):
        self.surrogate = surrogate
        self.acquisition = acquisition

    @abstractmethod
    def optimize(self, X: NDArray, bounds: Sequence[Tuple], param_types: Sequence[ParamType]) -> NDArray:
        pass


class RandomAcquisitionOpt(AcquisitionOptimizer):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction, rand: Optional[RandomState], sample_size=10):
        super(RandomAcquisitionOpt, self).__init__(surrogate, acquisition)
        self.rand = rand if rand else np.random
        self.sample_size = sample_size


    def optimize(self, X: NDArray, bounds: Sequence[Tuple], param_types: Sequence[ParamType]) -> NDArray:
        assert len(bounds) == len(param_types), "Must provide a ParamType and bound for each parameter."

        # Find best surrogate score so far
        yhat = self.surrogate.predict(X)
        best = yhat.min()
        n_params = len(bounds)
        X_samp = []

        # random search of the domain
        for _ in range(self.sample_size):
            x_i = np.array([self.rand_sample_in_bounds(bounds[j], param_types[j]) for j in range(n_params)])
            X_samp.append(x_i)
        
        X_samp: NDArray = np.array(X_samp)

        # find acquisition function value for each sample
        scores: NDArray = self.acquisition.acquire(self.surrogate, X_samp, best)

        return X_samp[np.argmin(scores)]


    def rand_sample_in_bounds(self, bounds: Tuple, param_type: ParamType) -> Union[float, int]:
        # Uniform sample
        if param_type == ParamType.Cont:
            x_j = self.rand.uniform(bounds[0], bounds[1])
        elif param_type == ParamType.Disc:
            x_j = self.rand.randint(bounds[0], bounds[1])
        return x_j
