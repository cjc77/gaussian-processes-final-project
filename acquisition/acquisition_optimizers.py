
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
    def optimize(self, X: NDArray) -> NDArray:
        pass


class RandomAcquisitionOpt(AcquisitionOptimizer):
    def __init__(self, surrogate: object, acquisition: AcquisitionFunction, rand: Optional[RandomState]):
        super(RandomAcquisitionOpt, self).__init__(surrogate, acquisition)
        self.rand = rand if rand else np.random
    
    def optimize(self, X: NDArray, bounds: NDArray) -> NDArray:
        yhat = self.surrogate.predict(X)
        best = yhat.min()

        # TODO - use bounds to randomly search the domain...

