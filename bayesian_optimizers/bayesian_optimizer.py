from abc import ABC, abstractmethod

from util.defs import *

class BayesianOptimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self, iterations, objective) -> Tuple[NDArray, NDArray]:
        pass
