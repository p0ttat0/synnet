from typing import Tuple
from axonlib.base.utility_layer_base import UtilityLayerBase
import numpy as np

class Pool(UtilityLayerBase):
    def __init__(self, kernel_size:int, stride:Tuple[int, int], padding: str, pool_mode="max"):
        self.stride = [stride, stride] if isinstance(stride, int) else stride
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        self.padding = (kernel_size-1, kernel_size-1) if padding=="full" else (0, 0)

        # --- For Backprop ---
        self.input_cache = None
        self.argmax_indexes = None

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        return input_tensor

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient