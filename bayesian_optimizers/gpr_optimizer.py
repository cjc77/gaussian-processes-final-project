from sklearn.gaussian_process import GaussianProcessRegressor

from util.defs import *
from bayesian_optimizers.bayesian_optimizer import BayesianOptimizer
from acquisition.acquisition_optimizers import AcquisitionOptimizer


class GPROptimizer():
    def __init__(self, gpr: GaussianProcessRegressor, X: NDArray, y: NDArray, opt_acquisition: AcquisitionOptimizer, objective: Objective, bounds: Sequence[Tuple], param_types: Sequence[ParamType], fit=False):
        self.gpr = gpr
        self.X = X
        self.y = y
        self.objective = objective
        self.opt_acquisition = opt_acquisition
        self.bounds = bounds
        self.param_types = param_types
        if fit:
            self.gpr.fit(X, y)

    def optimize(self, iterations: int, thresh: float = None) -> Tuple[NDArray, NDArray]:
        for _ in range(iterations):
            # Update model
            self.gpr.fit(self.X, self.y)

            # Select next point
            x = self.opt_acquisition.optimize(self.X, self.bounds, self.param_types)

            # Sample objective for new point
            yhat = self.objective(x)

            # Update dataset
            self.X = np.vstack((self.X, x))
            self.y = np.vstack((self.y, yhat))

            if thresh and yhat <= thresh:
                break
        
        return np.argmin(self.y)
