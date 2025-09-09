import numpy as np


class LossFunctionBase:
    """
    Base class for loss functions.

    This class defines the interface for loss functions. Subclasses should
    implement the `get_loss` and `get_d_loss` methods.
    """

    @staticmethod
    def get_loss(labels: np.ndarray, predictions: np.ndarray, batch_size: int) -> np.floating:
        """
        Computes the loss.

        Args:
            labels: The true labels.
            predictions: The predicted labels.
            batch_size: The batch size.

        Returns:
            The computed loss.
        """
        raise NotImplementedError("get_loss must be overridden in subclass")

    @staticmethod
    def get_d_loss(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss.

        Args:
            labels: The true labels.
            predictions: The predicted labels.

        Returns:
            The derivative of the loss.
        """
        raise NotImplementedError("get_d_loss must be overridden in subclass")
