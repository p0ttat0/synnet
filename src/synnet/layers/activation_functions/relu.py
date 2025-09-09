from synnet.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Relu(ActivationFunctionBase):
    """
    The rectified linear unit (ReLU) activation function.

    This class implements the ReLU activation function, which is defined as
    f(x) = max(0, x).
    """

    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        """
        Computes the ReLU activation function.

        Args:
            x: The input array.

        Returns:
            The output of the ReLU activation function.
        """
        return np.maximum(0, x).astype(x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the ReLU activation function.

        Args:
            x: The input array.

        Returns:
            The derivative of the ReLU activation function.
        """
        return (x > 0).astype(x.dtype)