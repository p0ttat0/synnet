from typing import Tuple
from axonlib.base.optimizer_base import OptimizerBase
import numpy as np


class NoOptimizer(OptimizerBase):
    """
    No optimizer.
    """
    def adjust_gradient(self, dw: np.ndarray, db: np.ndarray, lr: np.floating) -> Tuple[np.ndarray, ...]:
        """
        calculates and applies weight and bias gradients using optimizer
        :param dw: weight gradients
        :param db: bias gradients
        :param lr: learning rate
        :return: weight and bias changes
        """
        return dw * lr, db * lr
