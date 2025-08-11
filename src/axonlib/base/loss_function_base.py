import numpy as np


class LossFunctionBase:
    @staticmethod
    def get_loss(labels: np.ndarray, predictions: np.ndarray, batch_size:int) -> np.floating:
        raise NotImplementedError("get_loss must be overridden in subclass")

    @staticmethod
    def get_d_loss(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        raise NotImplementedError("get_d_loss must be overridden in subclass")