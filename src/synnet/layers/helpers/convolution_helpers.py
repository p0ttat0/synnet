from typing import Tuple
from numpy.lib.stride_tricks import as_strided
import numpy as np


def get_windows(input_tensor: np.ndarray, kernel_shape: Tuple[int, int, int, int], stride: Tuple[int, int],
                writable=False):
    """
    Extracts sliding windows from an input tensor for convolution.

    Args:
        input_tensor: The input tensor, with shape (batch, height, width,
            channels_in).
        kernel_shape: The shape of the kernel, (kernel_h, kernel_w,
            channels_in, channels_out).
        stride: The stride of the convolution, (stride_h, stride_w).
        writable: Whether the returned windows should be writable.

    Returns:
        An array of windows, with shape (batch, out_h, out_w, kernel_h,
        kernel_w, channels_in).
    """
    batch_size, height, width, input_channels = input_tensor.shape
    kernel_height, kernel_width, _, output_channels = kernel_shape

    output_height = 1 + (height - kernel_height) // stride[0]
    output_width = 1 + (width - kernel_width) // stride[1]

    batch_stride, height_stride, width_stride, channel_stride = input_tensor.strides
    strides = (
        batch_stride,
        height_stride * stride[0],
        width_stride * stride[1],
        height_stride,
        width_stride,
        channel_stride
    )

    windows = as_strided(
        input_tensor,
        shape=(batch_size, output_height, output_width, kernel_height, kernel_width, input_channels),
        strides=strides,
        writeable=writable
    )

    return windows


def pad(x: np.ndarray, padding: Tuple[int, int] | Tuple[int, int, int, int]):
    """
    Applies zero padding to a tensor.

    Args:
        x: The input tensor, with shape (batch_size, height, width, channels).
        padding: The padding to apply, either (height_padding, width_padding) or
            (top_padding, bottom_padding, left_padding, right_padding).

    Returns:
        The padded tensor.
    """
    if len(padding) == 2:
        pad_h, pad_w = padding
        bs, in_h, in_w, ch = x.shape

        padded = np.zeros((bs, (in_h + 2 * pad_h), (in_w + 2 * pad_w), ch), dtype=x.dtype)
        padded[:, pad_h:pad_h + in_h, pad_w:pad_w + in_w, :] = x

        return padded
    elif len(padding) == 4:
        pad_t, pad_b, pad_l, pad_r = padding
        bs, in_h, in_w, ch = x.shape
        padded = np.zeros((bs, (in_h + pad_t + pad_b), (in_w + pad_l + pad_r), ch), dtype=x.dtype)
        padded[:, pad_t:pad_t + in_h, pad_l:pad_l + in_w, :] = x

        return padded
    else:
        raise ValueError("padding must be a tuple of length 2 or 4")


def dilate(input_tensor: np.ndarray, dilation_rate: Tuple[int, int]):
    """
    Dilates a tensor.

    Args:
        input_tensor: The input tensor.
        dilation_rate: The dilation rate, (height_dilation, width_dilation).

    Returns:
        The dilated tensor.
    """
    if dilation_rate == (1, 1):
        return input_tensor

    output_height = (input_tensor.shape[1] - 1) * dilation_rate[0] + 1
    output_width = (input_tensor.shape[2] - 1) * dilation_rate[1] + 1
    dilated = np.zeros((input_tensor.shape[0], output_height, output_width, input_tensor.shape[3]))
    dilated[:, ::dilation_rate[1], ::dilation_rate[0], :] = input_tensor

    return dilated


def cross_correlate(input_tensor: np.ndarray, kernel: np.ndarray, stride: Tuple[int, int],
                    padding: Tuple[int, int] | Tuple[int, int, int, int]):
    """
    Performs cross-correlation.

    Args:
        input_tensor: The input tensor.
        kernel: The kernel.
        stride: The stride.
        padding: The padding.

    Returns:
        The result of the cross-correlation.
    """
    padded_input = pad(input_tensor, padding)

    # --- Dimensions ---
    batch_size, _, _, input_channels = input_tensor.shape
    kernel_height, kernel_width, _, output_channels = kernel.shape

    # --- Output Dimensions ---
    _, padded_height, padded_width, _ = padded_input.shape
    output_height = 1 + (padded_height - kernel_height) // stride[0]
    output_width = 1 + (padded_width - kernel_width) // stride[1]

    windows = get_windows(padded_input, kernel.shape, stride)

    x_col = np.reshape(windows, (batch_size * output_height * output_width, kernel_height * kernel_width * input_channels),
                       order='C')
    w_col = np.reshape(kernel, (kernel_height * kernel_width * input_channels, output_channels), order='F')
    output = np.dot(x_col, w_col)

    return output.reshape(batch_size, output_height, output_width, output_channels)


def convolve(input_tensor: np.ndarray, kernel: np.ndarray, stride: Tuple[int, int],
             padding: Tuple[int, int] | Tuple[int, int, int, int]):
    """
    Performs convolution.

    Args:
        input_tensor: The input tensor.
        kernel: The kernel.
        stride: The stride.
        padding: The padding.

    Returns:
        The result of the convolution.
    """
    return cross_correlate(input_tensor, np.rot90(kernel, 2), stride, padding)
