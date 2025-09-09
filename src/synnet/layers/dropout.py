from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
import numpy as np


class Dropout(UtilityLayerBase):
    """
    A dropout layer.

    This layer randomly sets a fraction of its inputs to zero during training,
    which helps to prevent overfitting.

    Attributes:
        dropout_rate (float): The fraction of inputs to set to zero.
        bin_map (np.ndarray): A binary map indicating which inputs to keep.
    """

    def __init__(self, dropout_rate: float):
        """
        Initializes the Dropout layer.

        Args:
            dropout_rate: The fraction of inputs to set to zero.
        """
        if dropout_rate > 1 or dropout_rate < 0:
            raise Exception("dropout rate must be between 0 and 1")
        self.bin_map = None
        self.dropout_rate = dropout_rate

    def param_init(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Initializes the parameters of the layer.

        Args:
            in_shape: The shape of the input to the layer.

        Returns:
            The shape of the output of the layer.
        """
        return in_shape

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        self.bin_map = np.random.rand(*input_tensor.shape[1:]) > self.dropout_rate
        return input_tensor * self.bin_map * 1 / (1 - self.dropout_rate)

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Performs the backward propagation step.

        Args:
            output_gradient: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input.
        """
        return output_gradient * self.bin_map * 1 / (1 - self.dropout_rate)