from synnet.base.loss_function_base import LossFunctionBase
import numpy as np


class CCE(LossFunctionBase):
    """
    Categorical cross-entropy loss function.

    This class implements the categorical cross-entropy loss function.
    """

    @staticmethod
    def get_loss(labels: np.ndarray, predictions: np.ndarray, batch_size: int) -> np.floating:
        """
        Computes the categorical cross-entropy loss.

        Args:
            labels: The true labels.
            predictions: The predicted labels.
            batch_size: The batch size.

        Returns:
            The categorical cross-entropy loss.
        """
        return -np.sum(labels * np.log(np.clip(predictions, 1e-7, 1 - 1e-7))) / batch_size

    @staticmethod
    def get_d_loss(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the categorical cross-entropy loss.

        Args:
            labels: The true labels.
            predictions: The predicted labels.

        Returns:
            The derivative of the categorical cross-entropy loss.
        """
        return predictions - labels
