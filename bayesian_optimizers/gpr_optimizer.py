from sklearn.gaussian_process import GaussianProcessRegressor
from numpy.random import RandomState

from util.defs import *
from bayesian_optimizers.bayesian_optimizer import BayesianOptimizer
from acquisition.acquisition_optimizers import AcquisitionOptimizer, random_x_sample


class GPROptimizer():
    def __init__(self, gpr: GaussianProcessRegressor, opt_acquisition: AcquisitionOptimizer, objective: Objective, bounds: NDArray, param_types: Sequence[ParamType], rand: Optional[RandomState], initial_samples=1, fit=False):
        """ 
        Args:
            gpr (GaussianProcessRegressor): An sklearn Gaussian process regressor.
            opt_acquisition (AcquisitionOptimizer): An acquisition optimizer.
            objective (Callable): An objective function that takes X as a parameter.
            bounds (NDArray): 2d array of bounds (column 0 for min, column 1 for max).
            param_types (Sequence): The types of each parameter (discrete, continuous).
            rand (Optional[RandomState]): Random state for seeding.
            initial_samples (int): Samples to take from objective function before 
                beginning optimization (>= 1).
            fit (bool): Fit Gaussian process regressor upon instantiation.
        """
        assert bounds.shape[1] == 2, "Bounds must be matrix of pairs of (max, min) values."
        assert initial_samples >= 1, "Need at least one initial sample from parameter domain."
        # need a random sample evaluated on the objective to start off
        X: NDArray = random_x_sample(bounds, param_types, samples=initial_samples, rand=rand)
        y: NDArray = np.array([objective(x) for x in X])
        if y.ndim == 1:
            y = y[:, np.newaxis]

        self.rand = rand
        self.gpr = gpr
        self.X = X
        self.y = y
        self.objective = objective
        self.opt_acquisition = opt_acquisition
        self.bounds = bounds
        self.param_types = param_types
        self.rand = rand
        if fit:
            self.gpr.fit(self.X, self.y)

    def optimize(self, iterations: int, thresh: float = None, verbose=False) -> Dict:
        """ Perform Bayesian optimization algorithm.
        
        Args:
            iterations (int): Max number of iterations before stopping.
            thresh (float): Objective return value which warrants stopping,
                i.e. a y value that is good enough to stop at.
            verbose (bool): Whether print out updates during optimization.
        
        Returns:
            Dict: A dictionary with the argmin index, x values which minimize
                the objective function, and minimum y value.
        """
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