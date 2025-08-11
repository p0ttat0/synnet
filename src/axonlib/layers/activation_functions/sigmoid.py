from axonlib.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Sigmoid(ActivationFunctionBase):
    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x, dtype=np.float64)).astype(x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        activated = 1 / (1 + np.exp(-x, dtype=np.float64)).astype(x.dtype)
        return activated * (1 - activated)
