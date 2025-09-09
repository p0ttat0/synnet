from synnet.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Sigmoid(ActivationFunctionBase):
    """
    The sigmoid activation function.

    This class implements the sigmoid activation function, which is defined as
    f(x) = 1 / (1 + exp(-x)).
    """

    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid activation function.

        Args:
            x: The input array.

        Returns:
            The output of the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x, dtype=np.float64)).astype(x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the sigmoid activation function.

        Args:
            x: The input array.

        Returns:
            The derivative of the sigmoid activation function.
        """
        activated = 1 / (1 + np.exp(-x, dtype=np.float64)).astype(x.dtype)
        return activated * (1 - activated)