from axonlib.base.activation_function_base import ActivationFunctionBase
import numpy as np

class Relu(ActivationFunctionBase):
    @staticmethod
    def default(x:np.ndarray) -> np.ndarray:
        return np.maximum(0, x).astype(x.dtype)

    @staticmethod
    def derivative(x:np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)
