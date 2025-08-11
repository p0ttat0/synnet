from typing import Tuple
import numpy as np


class LearnableLayerBase:
    """
    Base class for learnable layers.

    :var dtype: numpy dtype
    :var weights: weights of the layer
    :var bias: bias of the layer
    """
    def __init__(self):
        self.dtype = np.float32
        self.weights = None
        self.bias = None

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError("forprop must be overridden in subclass")

    def backprop(self, output_gradient: np.ndarray, lr: float, clip_value: float) -> np.ndarray:
        raise NotImplementedError("backprop must be overridden in subclass")

    def param_init(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError("param_init must be overridden in subclass")
