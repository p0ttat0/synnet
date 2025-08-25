from typing import Tuple
from numpy.lib.stride_tricks import as_strided
import numpy as np

def get_windows(input_tensor: np.ndarray, kernel_shape: Tuple[int, int, int, int], stride: Tuple[int, int], writable=False):
    """
    Extracts sliding windows from input tensor for convolution. (NHWC)

    :param input_tensor: Shape (batch, height, width, channels_in).
    :param kernel_shape: Tuple (kernel_h, kernel_w, channels_in, channels_out).
    :param stride: Tuple [stride_h, stride_w].
    :param writable: whether windows are to be writable

    :returns: np.ndarray: Windows of shape (batch, out_h, out_w, kernel_h, kernel_w, channels_in).
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
    Zeros padding (NHWC)

    :param x: (batch_size, height, width, channels) np array
    :param padding: (height_padding, width_padding) or (top_padding, bottom_padding, left_padding, right_padding)

    :return: padded input (NHWC)
    """
    if len(padding) == 2:
        pad_h, pad_w = padding
        bs, in_h, in_w, ch = x.shape

        padded = np.zeros((bs, (in_h + 2 * pad_h), (in_w + 2 * pad_w), ch), dtype=x.dtype)
        padded[:, pad_h:pad_h+in_h, pad_w:pad_w+in_w, :] = x

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
    matrix dilation (NHWC)

    :param input_tensor: (NHWC)
    :param dilation_rate: (height_dilation, width_dilation)

    :return: dilated input (NHWC)
    """
    if dilation_rate == (1, 1): return input_tensor

    output_height = (input_tensor.shape[1] - 1) * dilation_rate[0] + 1
    output_width = (input_tensor.shape[2] - 1) * dilation_rate[1] + 1
    dilated = np.zeros((input_tensor.shape[0], output_height, output_width, input_tensor.shape[3]))
    dilated[:, ::dilation_rate[1], ::dilation_rate[0], :] = input_tensor

    return dilated

def cross_correlate(input_tensor: np.ndarray, kernel: np.ndarray, stride: Tuple[int, int], padding: Tuple[int, int] | Tuple[int, int, int, int]):
    """
    cross correlation (NHWC)

    :param input_tensor  : (NHWC)
    :param kernel : (kernel_height, kernel_width, input_channels, output_channels)
    :param stride : (vertical_stride, horizontal_stride)
    :param padding: (height_padding, width_padding) or (top_padding, bottom_padding, left_padding, right_padding)

    :returns: cross correlation result (NHWC)
    """

    padded_input = pad(input_tensor, padding)
    kernel = kernel

    # --- Dimensions ---
    batch_size, _, _, input_channels = input_tensor.shape
    kernel_height, kernel_width, _, output_channels = kernel.shape

    # --- Output Dimensions ---
    _, padded_height, padded_width, _ = padded_input.shape
    output_height = 1 + (padded_height - kernel_height) // stride[0]
    output_width = 1 + (padded_width - kernel_width) // stride[1]

    windows = get_windows(padded_input, kernel.shape, stride)

    x_col = np.reshape(windows, (batch_size * output_height * output_width, kernel_height * kernel_width * input_channels), order='C')
    w_col = np.reshape(kernel, (kernel_height * kernel_width * input_channels, output_channels), order='F')
    output = np.dot(x_col, w_col)

    return output.reshape(batch_size, output_height, output_width, output_channels)

def convolve(input_tensor: np.ndarray, kernel: np.ndarray, stride: Tuple[int, int], padding: Tuple[int, int] | Tuple[int, int, int, int]):
    """
    convolution built on cross_correlate (NHWC)

    :param input_tensor  : (NHWC)
    :param kernel : (kernel_height, kernel_width, input_channels, output_channels)
    :param stride : (vertical_stride, horizontal_stride)
    :param padding: (height_padding, width_padding) or (top_padding, bottom_padding, left_padding, right_padding)

    :return: convolution result (NHWC)
    """
    return cross_correlate(input_tensor, np.rot90(kernel, 2), stride, padding)