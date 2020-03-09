from sklearn.gaussian_process import GaussianProcessRegressor

from util.defs import *
from bayesian_optimizers.bayesian_optimizer import BayesianOptimizer
from acquisition.acquisition_optimizers import AcquisitionOptimizer


class GPROptimizer():
    def __init__(self, gpr: GaussianProcessRegressor, X: NDArray, y: NDArray, opt_acquisition: AcquisitionOptimizer, objective: Objective, bounds: NDArray, param_types: Sequence[ParamType], fit=False):
        assert bounds.shape[1] == 2, "Bounds must be matrix of pairs of (max, min) values."
        self.gpr = gpr
        self.X = X
        self.y = y
        self.objective = objective
        self.opt_acquisition = opt_acquisition
        self.bounds = bounds
        self.param_types = param_types
        if fit:
            self.gpr.fit(X, y)

    def optimize(self, iterations: int, thresh: float = None, verbose=False) -> Tuple[NDArray, NDArray]:
        for i in range(iterations):
            if verbose:
                print(f"Optimization iteration {i + 1}")
            # Update model
            self.gpr.fit(self.X, self.y)

            # Select next point
            x = self.opt_acquisition.optimize(self.X, self.bounds, self.param_types)
            if verbose:
                print(f"Selected next parameter sample from acquisition optimizer: {x}")

            # Sample objective for new point
            yhat = self.objective(x)
            if verbose:
                print(f"Objective value at sample: {np.round(yhat, decimals=4)}")
                print(f"==============================================================\n")

            # Update dataset
            self.X = np.vstack((self.X, x))
            self.y = np.vstack((self.y, yhat))

            if thresh and yhat <= thresh:
                break

        min_idx = np.argmin(self.y)
        res = {"argmin": min_idx, "minimizer": self.X[min_idx], "minimum": self.y[min_idx]}

        if verbose:
            print(f"Optimization yielded: {res}")

        return res