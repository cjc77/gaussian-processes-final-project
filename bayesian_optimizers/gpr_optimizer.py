from sklearn.gaussian_process import GaussianProcessRegressor

from util.defs import *
from bayesian_optimizer import BayesianOptimizer


class GPROptimizer():
    def __init__(self, gpr: GaussianProcessRegressor):
        self.gpr = gpr

    def optimize(self, iterations: int, objective: Callable[[NDArray], NDArray], acquisition: Callable[[GaussianProcessRegressor], NDArray], X: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        for _ in range(iterations):
            # Update model
            self.gpr.fit(X, y)

            # Select next point
            x = acquisition(self.gpr)

            # Sample objective for new point
            yhat = objective(x)

            # Update dataset
            X = np.vstack((X, x))
            y = np.vstack((y, yhat))
        
        return np.argmin(y)
