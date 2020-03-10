import numpy as np
from numpy.random import RandomState

from util.defs import *


def random_x_sample(bounds: NDArray, param_types: Sequence[ParamType], samples: int, rand: RandomState=None) -> NDArray:
    assert bounds.ndim == 2, f"Bounds must be 2d, not {bounds.shape}"
    assert len(bounds) == len(param_types), "Must have bounds and param types for each param."
    if not rand:
        rand = np.random
    if type(param_types) != NDArray:
        pt = np.array(param_types)

    X = rand.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(samples, len(bounds)))

    disc_inds = np.where(pt == ParamType.Disc)[0]
    X[:, disc_inds] = np.round(X[:, disc_inds])

    return X
