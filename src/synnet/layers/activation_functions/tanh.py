from synnet.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Tanh(ActivationFunctionBase):
    """
    The hyperbolic tangent (tanh) activation function.

    This class implements the tanh activation function, which is defined as
    f(x) = tanh(x).
    """

    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        """
        Computes the tanh activation function.

        Args:
            x: The input array.

        Returns:
            The output of the tanh activation function.
        """
        return np.tanh(x, dtype=x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the tanh activation function.

        Args:
            x: The input array.

        Returns:
            The derivative of the tanh activation function.
        """
        activated = np.tanh(x, dtype=x.dtype)
        return 1 - np.square(activated, dtype=x.dtype)