from typing import Tuple
from synnet.base.learnable_layer_base import LearnableLayerBase
from synnet.layers.activation_functions import Relu, Sigmoid, Tanh, Swish, Softmax
from synnet.layers.helpers.convolution_helpers import cross_correlate, convolve, dilate, pad
from synnet.layers.helpers.parameter_init import WeightInit, BiasInit
import numpy as np


class Conv(LearnableLayerBase):
    """
    A 2D convolutional layer.

    This layer performs a 2D convolution operation on the input tensor.

    Attributes:
        dtype (np.dtype): The data type of the layer's parameters.
        kernel_params (Tuple[int, int, int]): The parameters of the kernel,
            (kernel_height, kernel_width, output_channels).
        act_func (str): The activation function to use.
        weights_init (str): The weight initialization method to use.
        bias_init (str): The bias initialization method to use.
        padding (Tuple[int, int]): The padding to apply to the input.
        stride (Tuple[int, int]): The stride of the convolution.
        optimizer (OptimizerBase): The optimizer to use for updating the layer's
            parameters.
        weights (np.ndarray): The weights of the layer.
        bias (np.ndarray): The bias of the layer.
        input_cache (np.ndarray): A cache of the input tensor for backpropagation.
        unactivated_output_cache (np.ndarray): A cache of the unactivated output
            tensor for backpropagation.
        padded_in_h (int): The height of the padded input.
        padded_in_w (int): The width of the padded input.
        cropped_padded_in_h (int): The height of the cropped and padded input.
        cropped_padded_in_w (int): The width of the cropped and padded input.
    """

    def __init__(self, kernel_params: Tuple[int, int, int], act_func='relu', padding='valid', stride=(1, 1),
                 weights_init='he', bias_init='zero', dtype=np.float32):
        """
        Initializes the Conv layer.

        Args:
            kernel_params: The parameters of the kernel, (kernel_height,
                kernel_width, output_channels).
            act_func: The activation function to use.
            padding: The padding to apply to the input, either 'valid' or 'full'.
            stride: The stride of the convolution.
            weights_init: The weight initialization method to use.
            bias_init: The bias initialization method to use.
            dtype: The data type of the layer's parameters.
        """
        if len(kernel_params) != 3:
            raise ValueError("conv layer kernel params must be (kernel_height, kernel_width, output_channels)")
        if not (isinstance(stride, int) or isinstance(stride, tuple)):
            raise ValueError("conv layer stride must be int or tuple of ints")
        if act_func not in ['relu', 'sigmoid', 'tanh', 'swish']:
            raise ValueError(f"conv layer '{act_func}' activation function is not supported")
        if not padding in ['valid', 'full']:
            raise ValueError("conv layer padding must be 'full' or 'valid'")
        if weights_init not in ['lecun', 'swish', 'he', 'xavier']:
            raise ValueError(f"conv layer '{weights_init}' weight initialization is not supported")
        if bias_init not in ['zero', 'zeros', 'none']:
            raise ValueError(f"conv layer '{bias_init}' bias initialization is not supported")

        super().__init__()

        self._ACTIVATION_MAP = {
            'relu': Relu(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'swish': Swish(),
            'softmax': Softmax()
        }

        # hyper params
        self.dtype = dtype
        self.kernel_params = kernel_params
        self.act_func = act_func
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.padding = (kernel_params[0] - 1, kernel_params[1] - 1) if padding == 'full' else (0, 0)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.optimizer = None

        # learnable params
        self.weights = None
        self.bias = None

        # cached
        self.input_cache = None
        self.unactivated_output_cache = None

        # deployment
        self.padded_in_h = None
        self.padded_in_w = None
        self.cropped_padded_in_h = None
        self.cropped_padded_in_w = None

    def param_init(self, in_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Initializes the parameters of the layer.

        Args:
            in_shape: The shape of the input to the layer.

        Returns:
            The shape of the output of the layer.
        """
        in_h, in_w, in_ch = in_shape
        out_h = 1 + (in_h + 2 * self.padding[0] - self.kernel_params[0]) // self.stride[0]
        out_w = 1 + (in_w + 2 * self.padding[1] - self.kernel_params[1]) // self.stride[1]
        out_ch = self.kernel_params[2]

        in_size = in_h * in_w * in_ch
        out_size = out_h * out_w * out_ch
        kernel_shape = (self.kernel_params[0], self.kernel_params[1], in_ch, self.kernel_params[2])

        if out_size <= 0:
            raise Exception("conv layer output size would be less or equal to 0")

        if self.weights is None:
            if self.weights_init in ("he", "kaiming"):
                self.weights = WeightInit.he(in_size, kernel_shape, self.dtype)
            elif self.weights_init in ("xavier", "glorot"):
                self.weights = WeightInit.xavier(in_size, out_size, kernel_shape, self.dtype)
            elif self.weights_init == "lecun":
                self.weights = WeightInit.lecun(in_size, kernel_shape, self.dtype)
            elif self.weights_init == "swish":
                self.weights = WeightInit.swish(in_size, kernel_shape, self.dtype)
            else:
                raise NotImplementedError(f"{self.weights_init} weight initialization not implemented")

        if self.bias is None:
            if self.bias_init in ("zero", "zeros", "none"):
                self.bias = BiasInit.zero((self.kernel_params[2],), self.dtype)
            else:
                raise NotImplementedError(f"{self.bias_init} bias initialization not implemented")

        self.padded_in_h = in_h + 2 * self.padding[0]
        self.padded_in_w = in_w + 2 * self.padding[1]
        self.cropped_padded_in_h = (out_h - 1) * self.stride[0] + kernel_shape[0]
        self.cropped_padded_in_w = (out_w - 1) * self.stride[1] + kernel_shape[1]

        return out_h, out_w, out_ch

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        input_tensor = input_tensor.astype(self.dtype)
        unactivated = cross_correlate(input_tensor, self.weights, self.stride, self.padding) + self.bias
        activated = self._ACTIVATION_MAP[self.act_func].default(unactivated)

        self.input_cache = input_tensor
        self.unactivated_output_cache = unactivated

        return activated

    def backprop(self, output_gradient: np.ndarray, lr: float, clip_value: float) -> np.ndarray:
        """
        Performs the backward propagation step.

        Args:
            output_gradient: The gradient of the loss with respect to the output.
            lr: The learning rate.
            clip_value: The value to clip the gradients at.

        Returns:
            The gradient of the loss with respect to the input.
        """
        assert self.unactivated_output_cache is not None
        assert self.input_cache is not None

        output_gradient = self.dtype(output_gradient)
        lr = self.dtype(lr)
        clip_value = self.dtype(clip_value)
        batch_size = output_gradient.shape[0]
        cropped_in = pad(self.input_cache, self.padding)[:, :self.cropped_padded_in_h, :self.cropped_padded_in_w,
                     :]  # ignores outer values missed due to stride to preserve correct dimensions

        dz = dilate(output_gradient * self._ACTIVATION_MAP[self.act_func].derivative(self.unactivated_output_cache),
                    self.stride)
        dw = cross_correlate(cropped_in.transpose(3, 1, 2, 0), dz.transpose(1, 2, 0, 3), stride=(1, 1),
                             padding=(0, 0)).transpose(1, 2, 0, 3) / batch_size
        db = np.sum(dz, axis=(0, 1, 2)) / batch_size

        di_conv_padding = (  # adjusts padding to avoid calculating input gradients for active and inactive padding
            self.weights.shape[0] - 1 - self.padding[0],
            self.weights.shape[0] - 1 - self.padding[0] + (self.padded_in_h - self.cropped_padded_in_h),
            self.weights.shape[1] - 1 - self.padding[1],
            self.weights.shape[1] - 1 - self.padding[1] + (self.padded_in_w - self.cropped_padded_in_w),
        )

        di = convolve(dz, self.weights.transpose(0, 1, 3, 2), stride=(1, 1), padding=di_conv_padding)

        weight_change, bias_change = self.optimizer.adjust_gradient(dw, db, lr)
        self.weights -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

        return di
