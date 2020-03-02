from sklearn.gaussian_process import GaussianProcessRegressor

from util.defs import *
from bayesian_optimizers.bayesian_optimizer import BayesianOptimizer


class GPROptimizer():
    def __init__(self, gpr: GaussianProcessRegressor, X: NDArray, y: NDArray, fit=False):
        self.gpr = gpr
        self.X = X
        self.y = y
        if fit:
            self.gpr.fit(X, y)

    def optimize(self, iterations: int, objective: Callable[[NDArray], NDArray], acquisition: Callable[[GaussianProcessRegressor, NDArray], NDArray]) -> Tuple[NDArray, NDArray]:
        for _ in range(iterations):
            # Update model
            self.gpr.fit(self.X, self.y)

            # Select next point
            x = acquisition(self.gpr, self.X)

            # Sample objective for new point
            yhat = objective(x)

            # Update dataset
            self.X = np.vstack((self.X, x))
            self.y = np.vstack((self.y, yhat))
        
        return np.argmin(self.y)
