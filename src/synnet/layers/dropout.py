from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
import numpy as np


class Dropout(UtilityLayerBase):
    """
    Dropout layer.
    :var dropout_rate: dropout rate
    :var bin_map: binary map
    """
    def __init__(self, dropout_rate: float):
        if dropout_rate > 1 or dropout_rate < 0:
            raise Exception("dropout rate must be between 0 and 1")
        self.bin_map = None
        self.dropout_rate = dropout_rate

    def param_init(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Parameter initialization.
        :param in_shape: input shape
        :return: output size
        """
        return in_shape

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward propagation.
        :param input_tensor: input tensor
        :return: output tensor
        """
        self.bin_map = np.random.rand(*input_tensor.shape[1:]) > self.dropout_rate
        return input_tensor * self.bin_map

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Backward propagation.
        :param output_gradient: output gradient
        :return: input gradient
        """
        return output_gradient * self.bin_map
