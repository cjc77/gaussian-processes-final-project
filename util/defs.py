import numpy as np
from typing import List, Dict, Tuple, Sequence, Callable, Optional, Union
from enum import Enum

NDArray = np.ndarray
Objective = Callable[[NDArray], float]

class ParamType(Enum):
    Cont = 0
    Disc = 1
