import numpy as np


class OptimizerBase:
    """
    Base class for optimizers.

    This class defines the interface for optimizers. Subclasses should
    implement the `adjust_gradient` method.
    """

    def adjust_gradient(self, dw: np.ndarray, db: np.ndarray, lr: np.floating):
        """
        Adjusts the gradients.

        Args:
            dw: The gradients of the weights.
            db: The gradients of the biases.
            lr: The learning rate.
        """
        raise NotImplementedError("adjust_gradient must be overridden in subclass")