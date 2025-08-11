from typing import Tuple
import numpy as np


class UtilityLayerBase:
    """
    Base class for utility layers.
    """
    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError("forprop must be overridden in subclass")

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError("backprop must be overridden in subclass")

    def param_init(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError("param_init must be overridden in subclass")