import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal

from util.defs import *


class AcquisitionFunction(ABC):
    """ Acquisition functions are used by an acquisition optimizer in order
    to optimize inputs to a surrogate function.

    Attributes:
        epsilon (float): Value used in order to avoid divide by zero
            error. Should be small (<= 1e-9)
    """
    def __init__(self, epsilon=1e-11):
        """ 
        Args:
            epsilon (float): Initialization value for epsilon.
        """
        self.epsilon = epsilon

    @abstractmethod
    def acquire(self, X: NDArray, surrogate: object, f_hat: float) -> NDArray:
        """ Evaluate the acquisition function.
        
        Args:
            X (NDArray): A sample of values on which to evaluate the surrogate.
            surrogate (object): The surrogate which models the true objective.
            f_hat (float): The best predicted value of the surrogate 
                found so far.
        
        Returns:
            NDArray: One acquisition function evaluation for each row of X
                (in order). Output will be scalar for 1d X input.
            
        """
        pass


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, epsilon: float = 1e-11):
        super(ProbabilityOfImprovement, self).__init__(epsilon)

    def acquire(self, X: NDArray, surrogate: object, f_hat: float) -> NDArray:
        # Turn 1d vector into 1 x d vector
        if X.ndim < 2:
            X = X[np.newaxis, :]
        mu, std = surrogate.predict(X, return_std=True)
        if mu.ndim > 1:
            mu = mu[:, 0]

        # Avoid divide by zero error
        std += self.epsilon

        return multivariate_normal.cdf((mu - f_hat) / std)


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, epsilon: float = 1e-11):
        super(ExpectedImprovement, self).__init__(epsilon)

    def acquire(self, X: NDArray, surrogate: object, f_hat: float) -> NDArray:
        # Turn 1d vector into 1 x d vector
        if X.ndim < 2:
            X = X[np.newaxis, :]
        mu, std = surrogate.predict(X, return_std=True)
        if mu.ndim > 1:
            mu = mu[:, 0]

        # Avoid divide by zero error
        std += self.epsilon
        
        diff_mu_fhat = mu - f_hat
        Z = (diff_mu_fhat) / std
        return (diff_mu_fhat) * multivariate_normal.cdf(Z) + std * multivariate_normal.pdf(Z)
