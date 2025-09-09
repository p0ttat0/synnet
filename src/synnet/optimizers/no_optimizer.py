from typing import Tuple
from synnet.base.optimizer_base import OptimizerBase
import numpy as np


class NoOptimizer(OptimizerBase):
    """
    A placeholder optimizer that does not perform any optimization.

    This class can be used when no optimization is desired.
    """

    def adjust_gradient(self, dw: np.ndarray, db: np.ndarray, lr: np.floating) -> Tuple[np.ndarray, ...]:
        """
        Returns the gradients scaled by the learning rate.

        Args:
            dw: The gradients of the weights.
            db: The gradients of the biases.
            lr: The learning rate.

        Returns:
            A tuple containing the scaled weight and bias gradients.
        """
        return dw * lr, db * lr