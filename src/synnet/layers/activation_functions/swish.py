from synnet.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Swish(ActivationFunctionBase):
    """
    The Swish activation function.

    This class implements the Swish activation function, which is defined as
    f(x) = x * sigmoid(x).
    """

    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        """
        Computes the Swish activation function.

        Args:
            x: The input array.

        Returns:
            The output of the Swish activation function.
        """
        return (x / (1 + np.exp(-x, dtype=np.float64))).astype(x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Swish activation function.

        Args:
            x: The input array.

        Returns:
            The derivative of the Swish activation function.
        """
        sig = (1 / (1 + np.exp(-x, dtype=np.float64))).astype(x.dtype)
        return sig * (1 + x.astype(x.dtype) * (1 - sig))