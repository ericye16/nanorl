"""Type definitions."""

from typing import Any, Dict, NamedTuple

import jax
import numpy as np

Params = Any
PRNGKey = jax.random.KeyArray

LogDict = dict[str, float]


class Transition(NamedTuple):
    observation: Dict[str, np.ndarray]
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_observation: Dict[str, np.ndarray]
