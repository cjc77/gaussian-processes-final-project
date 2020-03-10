from abc import ABC, abstractmethod
from numpy.random import RandomState
from acquisition.acquisition_optimizers import random_x_sample
from sklearn.gaussian_process import GaussianProcessRegressor
from dask import compute

from util.defs import *
from util.util import random_x_sample, scalar_or_1d_to_2d, singleton_to_scalar
from acquisition.acquisition_optimizers import AcquisitionOptimizer


class HPOptimizer(ABC):
    def __init__(self, objective: Objective, bounds: NDArray, param_types: Sequence[ParamType], rand: Optional[RandomState]):
        """ 
        Args:
            objective (Callable): An objective function that takes X as a parameter.
            bounds (NDArray): 2d array of bounds (column 0 for min, column 1 for max).
            param_types (Sequence): The types of each parameter (discrete, continuous).
            rand (Optional[RandomState]): Random state for seeding.
        """
        assert bounds.shape[1] == 2, "Bounds must be matrix of pairs of (max, min) values."
        self.bounds: NDArray = bounds
        self.param_types: Sequence[ParamType] = param_types
        self.objective: Objective = objective
        self.rand: Optional[RandomState] = rand if rand else None
        self.X: NDArray = np.array([])
        self.y: NDArray = np.array([])

    @abstractmethod
    def optimize(self, iterations, objective) -> Dict:
        pass


class GPROptimizer(HPOptimizer):
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
        assert initial_samples >= 1, "Need at least one initial sample from parameter domain."

        super(GPROptimizer, self).__init__(objective, bounds, param_types, rand)

        # need a random sample evaluated on the objective to start off
        X: NDArray = random_x_sample(bounds, param_types, samples=initial_samples, rand=rand)
        y: NDArray = np.array([objective(x) for x in X])
        y = scalar_or_1d_to_2d(y)

        self.X: NDArray = X
        self.y: NDArray = y
        self.gpr = gpr
        self.opt_acquisition = opt_acquisition
        if fit:
            self.gpr.fit(self.X, self.y)

    def optimize(self, iterations: int, thresh: Optional[float] = None, verbose=False) -> Dict:
        """ Perform Bayesian optimization algorithm.
        
        Args:
            iterations (int): Max number of iterations before stopping.
            thresh (Optional[float]): Objective return value which warrants stopping,
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
            x: NDArray = self.opt_acquisition.optimize(self.X, self.bounds, self.param_types)
            if verbose:
                print(f"Selected next parameter sample from acquisition optimizer: {x}")

            # Sample objective for new point
            yhat = self.objective(x)
            if verbose:
                print(f"Objective value at sample: {np.round(yhat, decimals=4)}")
                print(f"==============================================================\n")

            # Update dataset
            self.X: NDArray = np.vstack((self.X, x))
            self.y: NDArray = np.vstack((self.y, yhat))

            if thresh and yhat <= thresh:
                break

        min_idx = np.argmin(self.y)

        res = {"argmin": min_idx, "minimizer": self.X[min_idx], "minimum": self.y[min_idx]}

        if verbose:
            print(f"Optimization yielded: {res}")

        return res


class RandomSearchOptimizer(HPOptimizer):
    def __init__(self, objective: Objective, bounds: NDArray, param_types: Sequence[ParamType], rand: Optional[RandomState], parallel=False):
        """ 
        Args:
            objective (Callable): An objective function that takes X as a parameter.
            bounds (NDArray): 2d array of bounds (column 0 for min, column 1 for max).
            param_types (Sequence): The types of each parameter (discrete, continuous).
            rand (Optional[RandomState]): Random state for seeding.
            parallel (bool): Whether to run random sampling in parallel.
        """
        super(RandomSearchOptimizer, self).__init__(objective, bounds, param_types, rand)
        self.parallel = parallel

    def optimize(self, iterations: int, thresh: Optional[float] = None, verbose=False) -> Dict:
        """ Perform random hyperparameter search.
        
        Args:
            iterations (int): Max number of iterations before stopping.
            thresh (Optional[float]): Objective return value which warrants stopping,
                i.e. a y value that is good enough to stop at. Not used if parallel
                set to True.
            verbose (bool): Whether print out updates during optimization.
        
        Returns:
            Dict: A dictionary with the argmin index, x values which minimize
                the objective function, and minimum y value.
        """
        if self.parallel:
            assert thresh is None, "Thresh cannot be set to  True if parallel being used."

        # choose random sample of y
        X: NDArray = random_x_sample(self.bounds, self.param_types, iterations, self.rand)

        if self.parallel:
            y: NDArray = self._opt_parallel(X, iterations, verbose)
        else:
            y: NDArray = self._opt_sync(X, iterations, thresh, verbose)
        
        if self.X.size == 0:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.vstack((self.y, y))

        min_idx = np.argmin(self.y)

        res = {"argmin": min_idx, "minimizer": self.X[min_idx], "minimum": self.y[min_idx]}

        if verbose:
            print(f"Optimization yielded: {res}")

        return res

    def _opt_parallel(self, X: NDArray, iterations: int, verbose: bool) -> NDArray:
        y = np.array(compute([self.objective(xi) for xi in X])[0])
        y = scalar_or_1d_to_2d(y)

        if verbose:
            for i in range(iterations):
                print(f"Optimization iteration {i + 1}")
                print(f"Selected next parameter sample: {X[i]}")
                print(f"Objective value at sample: {np.round(y[i, 0], decimals=4)}")
                print(f"==============================================================\n")

        return y

    def _opt_sync(self, X: NDArray, iterations: int, thresh: Optional[float], verbose: bool) -> NDArray:
        ys = []
        for i in range(iterations):
            if verbose:
                print(f"Optimization iteration {i + 1}")
                print(f"Selected next parameter sample: {X[i]}")

            # Sample objective for new point
            y = self.objective(X[i])
            y = singleton_to_scalar(y)
            if verbose:
                print(f"Objective value at sample: {np.round(y, decimals=4)}")
                print(f"==============================================================\n")

            ys.append(y)

            if thresh and y <= thresh:
                break
        
        return scalar_or_1d_to_2d(np.array(ys))
