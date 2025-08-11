from axonlib.base.loss_function_base import LossFunctionBase
import numpy as np


class MSE(LossFunctionBase):
    @staticmethod
    def get_loss(labels: np.ndarray, predictions: np.ndarray, batch_size: int) -> np.floating:
        return np.sum((labels-predictions)**2)/batch_size/2

    @staticmethod
    def get_d_loss(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return predictions - labels