from axonlib.base.activation_function_base import ActivationFunctionBase
import numpy as np


class Swish(ActivationFunctionBase):
    @staticmethod
    def default(x: np.ndarray) -> np.ndarray:
        return (x / (1 + np.exp(-x, dtype=np.float64))).astype(x.dtype)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        sig = (1 / (1 + np.exp(-x, dtype=np.float64))).astype(x.dtype)
        return sig * (1 + x.astype(x.dtype) * (1 - sig))
