from abc import ABC, abstractmethod

from util.defs import *

class BayesianOptimizer(ABC):
    @abstractmethod
    def optimize(self, iterations, objective) -> Tuple[NDArray, NDArray]:
        pass
