from typing import Tuple
from synnet.base.utility_layer_base import UtilityLayerBase
from synnet.layers.helpers.convolution_helpers import get_windows, pad
import numpy as np


class Pool(UtilityLayerBase):
    """
    A pooling layer.

    This layer performs a pooling operation on the input tensor.

    Attributes:
        stride (Tuple[int, int]): The stride of the pooling operation.
        kernel_size (int): The size of the pooling kernel.
        pool_mode (str): The pooling mode to use, either 'max' or 'average'.
        padding (Tuple[int, int]): The padding to apply to the input.
        input_cache (np.ndarray): A cache of the input tensor for backpropagation.
        normalized_mask (np.ndarray): A normalized mask for max pooling.
    """

    def __init__(self, kernel_size: int, stride: Tuple[int, int] | int, padding: str, pool_mode="max"):
        """
        Initializes the Pool layer.

        Args:
            kernel_size: The size of the pooling kernel.
            stride: The stride of the pooling operation.
            padding: The padding to apply to the input, either 'valid' or 'full'.
            pool_mode: The pooling mode to use, either 'max' or 'average'.
        """
        if kernel_size <= 0:
            raise ValueError("kernel sie must be greater than zero")
        if padding not in ["valid", "full"]:
            raise ValueError("padding must be valid or full")
        if pool_mode not in ["max", "average"]:
            raise ValueError("pool_mode must be max or average")

        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        self.padding = (kernel_size - 1, kernel_size - 1) if padding == "full" else (0, 0)

        # --- For Backprop ---
        self.input_cache = None
        self.normalized_mask = None

    def param_init(self, in_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Initializes the parameters of the layer.

        Args:
            in_shape: The shape of the input to the layer.

        Returns:
            The shape of the output of the layer.
        """
        in_h, in_w, in_ch = in_shape
        out_h = 1 + (in_h + 2 * self.padding[0] - self.kernel_size) // self.stride[0]
        out_w = 1 + (in_w + 2 * self.padding[1] - self.kernel_size) // self.stride[1]

        return out_h, out_w, in_ch

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        bs, in_h, in_w, in_ch = input_tensor.shape
        windows = get_windows(pad(input_tensor, self.padding), (self.kernel_size, self.kernel_size, in_ch, in_ch),
                              self.stride)

        if self.pool_mode == "max":
            max_val = windows.max(axis=(3, 4), keepdims=True)
            max_mask = (windows == max_val)

            self.normalized_mask = max_mask / np.sum(max_mask, axis=(3, 4), keepdims=True)
            self.input_cache = input_tensor

            return max_val.squeeze(axis=(3, 4))
        elif self.pool_mode == "average":
            self.input_cache = input_tensor

            return windows.mean(axis=(3, 4))

        raise NotImplementedError(f"{self.pool_mode} pooling not implemented")

    def backprop(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Performs the backward propagation step.

        Args:
            output_gradient: The gradient of the loss with respect to the output.

        Returns:
            The gradient of the loss with respect to the input.
        """
        assert self.input_cache is not None

        bs, in_h, in_w, in_ch = self.input_cache.shape
        di = pad(np.zeros_like(self.input_cache), self.padding)
        di_windows = get_windows(di, (self.kernel_size, self.kernel_size, in_ch, in_ch), self.stride,
                                 writable=True)  # di_windows points directly to memory locations of elements in di

        grad_expanded = output_gradient[:, :, :, np.newaxis, np.newaxis, :]

        if self.pool_mode == "max":
            # broadcasts gradient with associated normalized window mask then adds that to the windows
            grad_windows = grad_expanded * self.normalized_mask
            np.add.at(di_windows, None, grad_windows)
            di = di[:, self.padding[0]:self.padding[0] + in_h, self.padding[1]:self.padding[1] + in_w, :]

            return di
        elif self.pool_mode == "average":
            # broadcast adds normalized gradient with associated window
            grad_windows = grad_expanded / self.kernel_size ** 2
            np.add.at(di_windows, None, grad_windows)
            di = di[:, self.padding[0]:self.padding[0] + in_h, self.padding[1]:self.padding[1] + in_w, :]

            return di

        raise NotImplementedError(f"{self.pool_mode} pooling not implemented")
