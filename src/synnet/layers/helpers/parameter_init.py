from typing import Tuple
import numpy as np


class WeightInit:
    """
    A wrapper class for common weight initialization techniques.

    This class provides static methods for initializing the weights of a layer
    using various techniques.
    """

    @staticmethod
    def lecun(in_size: int, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Initializes weights using the LeCun initialization method.

        This method is recommended for SELU or tanh activation functions.

        Args:
            in_size: The number of input units.
            shape: The shape of the weight tensor.
            dtype: The data type of the weights.

        Returns:
            The initialized weight tensor.
        """
        return np.random.normal(scale=np.sqrt(1 / in_size), size=shape).astype(dtype)

    @staticmethod
    def swish(in_size: int, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Initializes weights using a method recommended for the Swish activation
        function.

        Args:
            in_size: The number of input units.
            shape: The shape of the weight tensor.
            dtype: The data type of the weights.

        Returns:
            The initialized weight tensor.
        """
        return np.random.normal(scale=np.sqrt(1.1 / in_size), size=shape).astype(dtype)

    @staticmethod
    def he(in_size: int, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Initializes weights using the He initialization method.

        This method is recommended for ReLU or ReLU-variant activation functions.

        Args:
            in_size: The number of input units.
            shape: The shape of the weight tensor.
            dtype: The data type of the weights.

        Returns:
            The initialized weight tensor.
        """
        return np.random.normal(scale=np.sqrt(1 / in_size), size=shape).astype(dtype)

    @staticmethod
    def xavier(in_size: int, out_size: int, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Initializes weights using the Xavier initialization method.

        This method is recommended for sigmoid or tanh activation functions.

        Args:
            in_size: The number of input units.
            out_size: The number of output units.
            shape: The shape of the weight tensor.
            dtype: The data type of the weights.

        Returns:
            The initialized weight tensor.
        """
        return np.random.normal(scale=np.sqrt(2 / in_size + out_size), size=shape).astype(dtype)


class BiasInit:
    """
    A wrapper class for common bias initialization techniques.

    This class provides static methods for initializing the biases of a layer.
    """

    @staticmethod
    def zero(shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Initializes biases to zero.

        Args:
            shape: The shape of the bias tensor.
            dtype: The data type of the biases.

        Returns:
            The initialized bias tensor.
        """
        return np.zeros(shape=shape, dtype=dtype)
