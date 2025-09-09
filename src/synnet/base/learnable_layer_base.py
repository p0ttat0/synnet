from typing import Tuple
import numpy as np


class LearnableLayerBase:
    """
    Base class for learnable layers.

    This class defines the interface for learnable layers, which are layers
    that have weights and biases that can be updated during training.
    """

    def __init__(self):
        """
        Initializes the LearnableLayerBase.
        """
        self.dtype = np.float32
        self.weights = None
        self.bias = None

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        raise NotImplementedError("forprop must be overridden in subclass")

    def backprop(self, output_gradient: np.ndarray, lr: float, clip_value: float) -> np.ndarray:
        """
        Performs the backward propagation step.

        Args:
            output_gradient: The gradient of the loss with respect to the output.
            lr: The learning rate.
            clip_value: The value to clip the gradients at.

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