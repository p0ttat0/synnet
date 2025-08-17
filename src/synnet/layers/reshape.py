from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
import numpy as np


class Reshape(UtilityLayerBase):
    """
    Reshape layer.

    :var input_shape: input shape
    :var output_shape: output shape
    """
    def __init__(self, output_shape: Tuple[int, ...]):
        self.output_shape = output_shape
        self.input_shape = None

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward propagation.
        :param input_tensor: input tensor
        :return: output tensor
        """
        return input_tensor.reshape((-1,) + self.output_shape)

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Backward propagation.
        :param output_gradient: output gradient
        :return: input gradient
        """
        return output_gradient.reshape((-1,) + self.input_shape)

    def param_init(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Parameter initialization.
        :param input_shape: input shape
        :return: output shape
        """
        self.input_shape = input_shape
        return self.output_shape
