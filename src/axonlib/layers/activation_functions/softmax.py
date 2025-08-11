from axonlib.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Softmax(ActivationFunctionBase):
    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True), dtype=np.float64).astype(x.dtype)
        return e_x / np.sum(e_x, axis=1, keepdims=True, dtype=x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x, dtype=x.dtype)
