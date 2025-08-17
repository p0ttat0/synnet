import numpy as np


class OptimizerBase:
    """
    Base class for optimizers.
    """
    def adjust_gradient(self, dw: np.ndarray, db: np.ndarray, lr: np.floating):
        """
        Adjusts the gradients.
        :param dw: weight gradients
        :param db: bias gradients
        :param lr: learning rate
        """
        raise NotImplementedError("adjust_gradient must be overridden in subclass")
