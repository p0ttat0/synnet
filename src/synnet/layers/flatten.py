from math import prod
from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
import numpy as np


class Flatten(UtilityLayerBase):
    """
    Flatten layer.

    :var input_shape: input shape
    :var output_shape: output shape
    """
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward propagation.
        :param input_tensor: input tensor
        :return: output tensor
        """
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Backward propagation.
        :param output_gradient: output gradient
        :return: input gradient
        """
        return output_gradient.reshape((-1, ) + self.input_shape)

    def param_init(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Parameter initialization.
        :param input_shape: input shape
        :return: output shape
        """
        self.input_shape = input_shape
        self.output_shape = (prod(input_shape),)
        return self.output_shape
