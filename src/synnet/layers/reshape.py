from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
import numpy as np


class Reshape(UtilityLayerBase):
    """
    A reshape layer.

    This layer reshapes the input tensor to the specified output shape.

    Attributes:
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
    """

    def __init__(self, output_shape: Tuple[int, ...]):
        """
        Initializes the Reshape layer.

        Args:
            output_shape: The desired output shape.
        """
        self.output_shape = output_shape
        self.input_shape = None

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        return input_tensor.reshape((-1,) + self.output_shape)

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
        return self.output_shape