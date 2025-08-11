import numpy as np


class ActivationFunctionBase:
    @staticmethod
    def default(x:np.ndarray) -> np.ndarray:
        raise NotImplementedError("default must be overridden in subclass")

    @staticmethod
    def derivative(x:np.ndarray) -> np.ndarray:
        raise NotImplementedError("derivative must be overridden in subclass")