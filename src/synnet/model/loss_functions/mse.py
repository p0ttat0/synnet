from synnet.base.loss_function_base import LossFunctionBase
import numpy as np


class MSE(LossFunctionBase):
    """
    Mean squared error loss function.

    This class implements the mean squared error loss function.
    """

    @staticmethod
    def get_loss(labels: np.ndarray, predictions: np.ndarray, batch_size: int) -> np.floating:
        """
        Computes the mean squared error loss.

        Args:
            labels: The true labels.
            predictions: The predicted labels.
            batch_size: The batch size.

        Returns:
            The mean squared error loss.
        """
        return np.sum((labels - predictions) ** 2) / batch_size / 2

    @staticmethod
    def get_d_loss(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the mean squared error loss.

        Args:
            labels: The true labels.
            predictions: The predicted labels.

        Returns:
            The derivative of the mean squared error loss.
        """
        return predictions - labels
