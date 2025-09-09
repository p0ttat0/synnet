from typing import Tuple
from synnet.base.optimizer_base import OptimizerBase
import numpy as np


class Adam(OptimizerBase):
    """
    The Adam optimizer.

    This class implements the Adam optimization algorithm.

    Attributes:
        step (int): The current optimization step.
        epsilon (float): A small value to prevent division by zero.
        b1 (float): The exponential decay rate for the first moment estimates.
        b2 (float): The exponential decay rate for the second moment estimates.
        fme (dict): The first moment estimates for the weights and biases.
        sme (dict): The second moment estimates for the weights and biases.
    """

    def __init__(self, b1: float = 0.9, b2: float = 0.999, epsilon: float = 1e-8):
        """
        Initializes the Adam optimizer.

        Args:
            b1: The exponential decay rate for the first moment estimates.
            b2: The exponential decay rate for the second moment estimates.
            epsilon: A small value to prevent division by zero.
        """
        self.step = 1
        self.epsilon = epsilon
        self.b1 = b1
        self.b2 = b2
        self.fme = {"weights": np.ndarray(0), "bias": np.ndarray(0)}
        self.sme = {"weights": np.ndarray(0), "bias": np.ndarray(0)}

    def adjust_gradient(self, dw: np.ndarray, db: np.ndarray, lr: np.floating) -> Tuple[np.ndarray, ...]:
        """
        Adjusts the gradients using the Adam optimization algorithm.

        Args:
            dw: The gradients of the weights.
            db: The gradients of the biases.
            lr: The learning rate.

        Returns:
            A tuple containing the adjusted weight and bias gradients.
        """
        self.fme["weights"] = (self.b1 * self.fme["weights"] + (1 - self.b1) * dw)
        self.fme["bias"] = (self.b1 * self.fme["bias"] + (1 - self.b1) * db)
        self.sme["weights"] = (self.b2 * self.sme["weights"] + (1 - self.b2) * np.square(dw))
        self.sme["bias"] = (self.b2 * self.sme["bias"] + (1 - self.b2) * np.square(db))

        bc_m_weights = self.fme["weights"] / (1 - self.b1 ** self.step)
        bc_v_weights = self.sme["weights"] / (1 - self.b2 ** self.step)
        bc_m_bias = self.fme["bias"] / (1 - self.b1 ** self.step)
        bc_v_bias = self.sme["bias"] / (1 - self.b2 ** self.step)

        new_dw = lr * bc_m_weights / (np.sqrt(bc_v_weights + self.epsilon))
        new_db = lr * bc_m_bias / (np.sqrt(bc_v_bias + self.epsilon))

        return new_dw, new_db
