import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm

from util.defs import *


class Acquisition(ABC):
    @abstractmethod
    def acquire(self, X: NDArray, f_hat: float) -> NDArray:
        pass


class ProbabilityOfImprovement(Acquisition):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def acquire(self, surrogate: object, X: NDArray, f_hat: float) -> NDArray:
        mu, std = surrogate.predict(X, return_std=True)
        mu = mu[:, 0]

        return norm.cdf((mu - f_hat) / std + self.epsilon)