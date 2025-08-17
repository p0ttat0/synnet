from typing import Tuple
from synnet.base.learnable_layer_base import LearnableLayerBase
from synnet.layers.activation_functions import Relu, Sigmoid, Tanh, Swish, Softmax
from synnet.layers.helpers.convolution_helpers import cross_correlate, convolve, dilate
from synnet.layers.helpers.parameter_init import WeightInit, BiasInit
import numpy as np


class Conv(LearnableLayerBase):
    """
    assumes inputs are (batchsize, input_height, input_width, input_channels) np array

    :var dtype: np dtype
    :var kernel_params: (kernel_height, kernel_width, output_channels)
    :var act_func: activation function (string)
    :var weights_init: weights initialization technique (string)
    :var bias_init: bias initialization technique (string)
    :var padding: padding (int)
    :var stride: stride (int)
    :var optimizer: optimizer obj
    :var stride: (height_stride, width_stride)
    :var padding: (v_padding, h_padding)
    :var weights: (kernel_height, kernel_width, input_channels, output_channels) np array
    :var bias: (output_channels, ) np array
    :var input_cache: (batchsize, input_height, input_width, input_channels) np array
    :var unactivated_output_cache: (batchsize, output_height, output_width, output_channels) np array
    :var cropped_input_width: the width of the active space in the input where the kernel actually touches
    :var cropped_input_height: the height of the active space in the input where the kernel actually touches
    """
    def __init__(self, kernel_params: Tuple[int, int, int], act_func='relu', padding='valid', stride=(1, 1), weights_init='he', bias_init='zero', dtype=np.float32):
        if len(kernel_params) != 3: raise ValueError("conv layer kernel params must be (kernel_height, kernel_width, output_channels)")
        if not (isinstance(stride, int) or isinstance(stride, tuple)): raise ValueError("conv layer stride must be int or tuple of ints")
        if act_func not in ['relu', 'sigmoid', 'tanh', 'swish']: raise  ValueError(f"conv layer '{act_func}' activation function is not supported")
        if not padding in ['valid', 'full']: raise ValueError("conv layer padding must be 'full' or 'valid'")
        if weights_init not in ['lecun', 'swish', 'he', 'xavier']: raise ValueError(f"conv layer '{weights_init}' weight initialization is not supported")
        if bias_init not in ['zero', 'zeros', 'none']: raise ValueError(f"conv layer '{bias_init}' bias initialization is not supported")

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
        self.padding = (kernel_params[0] - 1, kernel_params[1] - 1) if padding=='full' else (0, 0)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.optimizer = None

        # learnable params
        self.weights = None
        self.bias = None

        # cached
        self.input_cache = None
        self.unactivated_output_cache = None

        # deployment
        self.cropped_input_height = None
        self.cropped_input_width = None

    def param_init(self, in_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        bias, weight and optimizer initialization
        :param in_shape: number of neurons in previous layer
        """
        in_h, in_w, in_ch = in_shape
        out_h = (in_h + 2*self.padding[0] - self.kernel_params[0] + 1) // self.stride[0]
        out_w = (in_w + 2*self.padding[1] - self.kernel_params[1] + 1) // self.stride[1]
        out_ch = self.kernel_params[2]

        in_size = in_h*in_w*in_ch
        out_size = out_h*out_w*out_ch
        kernel_shape = (self.kernel_params[0], self.kernel_params[1], in_ch, self.kernel_params[2])

        if out_size <= 0:
            raise Exception("conv layer output size would be less or equal to 0")

        if self.weights is None:
            match self.weights_init:
                case "he" | "kaiming":
                    self.weights = WeightInit.he(in_size, kernel_shape, self.dtype)
                case "xavier" | "glorot":
                    self.weights = WeightInit.xavier(in_size, out_size, kernel_shape, self.dtype)
                case "lecun":
                    self.weights = WeightInit.lecun(in_size, kernel_shape, self.dtype)
                case "swish":
                    self.weights = WeightInit.swish(in_size, kernel_shape, self.dtype)
                case _:
                    raise NotImplementedError(f"{self.weights_init} weight initialization not implemented")

        if self.bias is None:
            match self.bias_init:
                case "zero" | "zeros"| "none":
                    self.bias = BiasInit.zero((self.kernel_params[2],), self.dtype)
                case _:
                    raise NotImplementedError(f"{self.bias_init} bias initialization not implemented")

        self.cropped_input_height = (out_h - 1) * self.stride[0] + kernel_shape[0]
        self.cropped_input_width = (out_w - 1) * self.stride[1] + kernel_shape[1]

        return out_h, out_w, out_ch

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        forward propagation + caches input (only inputs not ignored by striding) and unactivated output
        :param input_tensor: (NHWC)
        :return: activated output (NHWC)
        """
        input_tensor = input_tensor.astype(self.dtype)
        unactivated = cross_correlate(input_tensor, self.weights, self.stride, self.padding) + self.bias
        activated = self._ACTIVATION_MAP[self.act_func].default(unactivated)

        self.input_cache = input_tensor
        self.unactivated_output_cache = unactivated

        return activated


    def backprop(self, output_gradient: np.ndarray, lr: float, clip_value: float) -> np.ndarray:
        assert self.unactivated_output_cache is not None
        assert self.input_cache is not None

        output_gradient = self.dtype(output_gradient)
        lr = self.dtype(lr)
        clip_value = self.dtype(clip_value)
        batch_size = output_gradient.shape[0]
        cropped_input_cache = self.input_cache[:, :self.cropped_input_height, :self.cropped_input_width, :]     # ignores outer values missed due to stride to preserve correct dimensions

        dz = dilate(output_gradient * self._ACTIVATION_MAP[self.act_func].derivative(self.unactivated_output_cache), self.stride)
        dw = cross_correlate(cropped_input_cache.transpose(3, 1, 2, 0), dz.transpose(1, 2, 0, 3), stride=(1, 1), padding=self.padding).transpose(1, 2, 0, 3)/batch_size
        db = np.sum(dz, axis=(0, 1, 2))/batch_size
        di = convolve(dz, self.weights.transpose(0, 1, 3, 2), stride=(1, 1), padding=(self.weights.shape[0]-1-self.padding[0], self.weights.shape[1]-1-self.padding[1]))

        weight_change, bias_change = self.optimizer.adjust_gradient(dw, db, lr)
        self.weights -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

        return di