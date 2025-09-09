from math import prod
from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
import numpy as np


class Flatten(UtilityLayerBase):
    """
    A flatten layer.

    This layer flattens the input tensor into a 2D tensor.

    Attributes:
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
    """

    def __init__(self):
        """
        Initializes the Flatten layer.
        """
        self.input_shape = None
        self.output_shape = None

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Performs the backward propagation step.

        Args:
            output_gradient: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input.
        """
        return output_gradient.reshape((-1,) + self.input_shape)

    def param_init(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Initializes the parameters of the layer.

        Args:
            input_shape: The shape of the input to the layer.

        Returns:
            The shape of the output of the layer.
        """
        self.input_shape = input_shape
        self.output_shape = (prod(input_shape),)
        return self.output_shape