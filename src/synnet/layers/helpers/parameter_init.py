from typing import Tuple
import numpy as np


class WeightInit:
    """
    wrapper class for common weight initialization techniques

    supports: lecun, swish, he, and xavier
    """
    @staticmethod
    def lecun(in_size:int, shape:Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """weight initialization method recommended for SELU or tanh"""
        return np.random.normal(scale=np.sqrt(1/in_size), size=shape).astype(dtype)

    @staticmethod
    def swish(in_size:int, shape:Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """weight initialization method recommended for swish"""
        return np.random.normal(scale=np.sqrt(1.1/in_size), size=shape).astype(dtype)

    @staticmethod
    def he(in_size:int, shape:Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """weight initialization method recommended for ReLU or ReLU variants"""
        return np.random.normal(scale=np.sqrt(1/in_size), size=shape).astype(dtype)

    @staticmethod
    def xavier(in_size:int, out_size:int, shape:Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """weight initialization method recommended for sigmoid or tanh"""
        return np.random.normal(scale=np.sqrt(2/in_size+out_size), size=shape).astype(dtype)

class BiasInit:
    """
    wrapper class for common bias initialization techniques

    supports: zero
    """
    @staticmethod
    def zero(shape:Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """zero bias initialization"""
        return np.zeros(shape=shape, dtype=dtype)