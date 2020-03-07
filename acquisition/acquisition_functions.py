import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal

from util.defs import *


class AcquisitionFunction(ABC):
    @abstractmethod
    def acquire(self, surrogate: object, X: NDArray, f_hat: float) -> NDArray:
        pass


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, epsilon: float = 1e-11):
        self.epsilon = epsilon

    def acquire(self, surrogate: object, X: NDArray, f_hat: float) -> NDArray:
        mu, std = surrogate.predict(X, return_std=True)
        if mu.ndim > 1:
            mu = mu[:, 0]

        # Avoid divide by zero error
        std += self.epsilon

        return multivariate_normal.cdf((mu - f_hat) / std)


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, epsilon: float = 1e-11):
        self.epsilon = epsilon

    def acquire(self, surrogate: object, X: NDArray, f_hat: float) -> NDArray:
        mu, std = surrogate.predict(X, return_std=True)
        if mu.ndim > 1:
            mu = mu[:, 0]

        # Avoid divide by zero error
        std += self.epsilon
        
        diff_mu_fhat = mu - f_hat
        Z = (diff_mu_fhat) / std
        return (diff_mu_fhat) * multivariate_normal.cdf(Z) + std * multivariate_normal.pdf(Z)
