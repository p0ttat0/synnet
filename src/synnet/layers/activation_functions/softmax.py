from synnet.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Softmax(ActivationFunctionBase):
    """
    The softmax activation function.

    This class implements the softmax activation function, which is defined as
    f(x_i) = exp(x_i) / sum(exp(x_j)) for j in range(len(x)).
    """

    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        """
        Computes the softmax activation function.

        Args:
            x: The input array.

        Returns:
            The output of the softmax activation function.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True), dtype=np.float64).astype(x.dtype)
        return e_x / np.sum(e_x, axis=1, keepdims=True, dtype=x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the softmax activation function.

        Args:
            x: The input array.

        Returns:
            The derivative of the softmax activation function.
        """
        return np.ones_like(x, dtype=x.dtype)