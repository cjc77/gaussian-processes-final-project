from abc import ABC, abstractmethod
from numpy.random import RandomState
from acquisition.acquisition_optimizers import random_x_sample

from util.defs import *


class HPOptimizer(ABC):
    def __init__(self, objective: Objective, bounds: NDArray, param_types: Sequence[ParamType], rand: Optional[RandomState], initial_samples=1):
        """ 
        Args:
            objective (Callable): An objective function that takes X as a parameter.
            bounds (NDArray): 2d array of bounds (column 0 for min, column 1 for max).
            param_types (Sequence): The types of each parameter (discrete, continuous).
            rand (Optional[RandomState]): Random state for seeding.
            initial_samples (int): Samples to take from objective function before 
                beginning optimization (>= 1).
        """
        assert bounds.shape[1] == 2, "Bounds must be matrix of pairs of (max, min) values."
        assert initial_samples >= 1, "Need at least one initial sample from parameter domain."
        # need a random sample evaluated on the objective to start off
        X: NDArray = random_x_sample(bounds, param_types, samples=initial_samples, rand=rand)
        y: NDArray = np.array([objective(x) for x in X])
        if y.ndim == 1:
            y = y[:, np.newaxis]

        self.X = X
        self.y = y
        self.bounds = bounds
        self.param_types = param_types
        self.objective = objective
        self.rand = rand if rand else np.random

    @abstractmethod
    def optimize(self, iterations, objective) -> Dict:
        pass
