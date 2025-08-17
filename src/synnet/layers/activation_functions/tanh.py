from synnet.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Tanh(ActivationFunctionBase):
    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        return np.tanh(x, dtype=x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        activated = np.tanh(x, dtype=x.dtype)
        return 1 - np.square(activated, dtype=x.dtype)
