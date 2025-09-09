import numpy as np


class ActivationFunctionBase:
    """
    Base class for activation functions.

    This class defines the interface for activation functions. Subclasses should
    implement the `default` and `derivative` methods.
    """

    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        """
        Computes the activation function.

        Args:
            x: The input array.

        Returns:
            The output of the activation function.
        """
        raise NotImplementedError("default must be overridden in subclass")

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function.

        Args:
            x: The input array.

        Returns:
            The derivative of the activation function.
        """
        raise NotImplementedError("derivative must be overridden in subclass")
