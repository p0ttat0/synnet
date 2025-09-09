from typing import Tuple
import numpy as np


class UtilityLayerBase:
    """
    Base class for utility layers.

    This class defines the interface for utility layers, which are layers
    that do not have weights or biases that can be updated during training.
    """

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        raise NotImplementedError("forprop must be overridden in subclass")

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Performs the backward propagation step.

        Args:
            output_gradient: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input.
        """
        raise NotImplementedError("backprop must be overridden in subclass")

    def param_init(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Initializes the parameters of the layer.

        Args:
            in_shape: The shape of the input to the layer.

        Returns:
            The shape of the output of the layer.
        """
        raise NotImplementedError("param_init must be overridden in subclass")
