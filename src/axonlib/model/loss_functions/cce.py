from axonlib.base.loss_function_base import LossFunctionBase
import numpy as np


class CCE(LossFunctionBase):
    @staticmethod
    def get_loss(labels: np.ndarray, predictions: np.ndarray, batch_size:int) -> np.floating:
        return -np.sum(labels * np.log(np.clip(predictions, 1e-7, 1 - 1e-7))) / batch_size

    @staticmethod
    def get_d_loss(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return predictions-labels