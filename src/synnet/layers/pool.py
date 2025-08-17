from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
from synnet.layers.helpers.convolution_helpers import get_windows, pad
import numpy as np


class Pool(UtilityLayerBase):
    """
    assumes inputs are (batchsize, input_height, input_width, input_channels) np array
    """
    def __init__(self, kernel_size:int, stride:Tuple[int, int] | int, padding: str, pool_mode="max"):
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        self.padding = (kernel_size-1, kernel_size-1) if padding=="full" else (0, 0)

        # --- For Backprop ---
        self.input_cache = None
        self.argmax_indexes = None

    def param_init(self, in_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        in_h, in_w, in_ch = in_shape
        out_h = (in_h + 2*self.padding[0] - self.kernel_size + 1) // self.stride[0]
        out_w = (in_w + 2*self.padding[1] - self.kernel_size + 1) // self.stride[1]
        return out_h, out_w, in_ch

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        bs, in_h, in_w, in_ch = input_tensor.shape
        windows = get_windows(pad(input_tensor, self.padding), (self.kernel_size, self.kernel_size, in_ch, in_ch), self.stride)
        bs, out_h, out_w, kernel_h, kernel_w, out_ch = windows.shape

        if self.pool_mode == "max":
            argmax = np.argmax(windows.reshape(bs, out_h, out_w, -1, out_ch), axis=3)
            print(argmax.shape, input_tensor.shape)
            asdasd
        return input_tensor

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient