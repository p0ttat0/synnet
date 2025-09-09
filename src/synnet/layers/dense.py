from typing import Tuple
from synnet.layers.activation_functions import Relu, Sigmoid, Tanh, Swish, Softmax
from synnet.layers.helpers.parameter_init import WeightInit, BiasInit
from synnet.base.learnable_layer_base import LearnableLayerBase
import numpy as np


class Dense(LearnableLayerBase):
    """
    A fully connected layer.

    This layer performs a linear transformation on the input tensor, followed by
    an activation function.

    Attributes:
        dtype (np.dtype): The data type of the layer's parameters.
        size (int): The number of neurons in the layer.
        act_func (str): The activation function to use.
        weights_init (str): The weight initialization method to use.
        bias_init (str): The bias initialization method to use.
        optimizer (OptimizerBase): The optimizer to use for updating the layer's
            parameters.
        weights (np.ndarray): The weights of the layer.
        bias (np.ndarray): The bias of the layer.
        input_cache (np.ndarray): A cache of the input tensor for backpropagation.
        unactivated_output_cache (np.ndarray): A cache of the unactivated output
            tensor for backpropagation.
    """

    def __init__(self, size: int, act_func="relu", weights_init="he", bias_init="zero", dtype=np.float32):
        """
        Initializes the Dense layer.

        Args:
            size: The number of neurons in the layer.
            act_func: The activation function to use.
            weights_init: The weight initialization method to use.
            bias_init: The bias initialization method to use.
            dtype: The data type of the layer's parameters.
        """
        if size <= 0:
            raise ValueError("size must be greater than zero")
        if act_func not in ['relu', 'sigmoid', 'tanh', 'swish', 'softmax']:
            raise ValueError(f"dense layer '{act_func}' activation function is not supported")
        if weights_init not in ['lecun', 'swish', 'he', 'xavier']:
            raise ValueError(f"dense layer '{weights_init}' weight initialization is not supported")
        if bias_init not in ['zero', 'zeros', 'none']:
            raise ValueError(f"dense layer '{bias_init}' bias initialization is not supported")

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
        self.size = size
        self.act_func = act_func
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.optimizer = None

        # learnable params
        self.weights = None
        self.bias = None

        # cached
        self.input_cache = None
        self.unactivated_output_cache = None

    def param_init(self, in_shape: Tuple[int]) -> Tuple[int]:
        """
        Initializes the parameters of the layer.

        Args:
            in_shape: The shape of the input to the layer.

        Returns:
            The shape of the output of the layer.
        """
        in_size = in_shape[0]
        out_size = self.size
        weights_shape = (in_size, out_size)

        if out_size <= 0:
            raise Exception("dense layer output size would be less or equal to 0")

        if self.weights is None:
            if self.weights_init in ("he", "kaiming"):
                self.weights = WeightInit.he(in_size, weights_shape, self.dtype)
            elif self.weights_init in ("xavier", "glorot"):
                self.weights = WeightInit.xavier(in_size, out_size, weights_shape, self.dtype)
            elif self.weights_init == "lecun":
                self.weights = WeightInit.lecun(in_size, weights_shape, self.dtype)
            elif self.weights_init == "swish":
                self.weights = WeightInit.swish(in_size, weights_shape, self.dtype)
            else:
                raise NotImplementedError(f"{self.weights_init} weight initialization not implemented")

        if self.bias is None:
            if self.bias_init in ("zero", "zeros", "none"):
                self.bias = BiasInit.zero((out_size,), self.dtype)
            else:
                raise NotImplementedError(f"{self.bias_init} bias initialization not implemented")

        return (out_size,)

    def forprop(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward propagation step.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        input_tensor = input_tensor.astype(self.dtype)
        unactivated = np.dot(input_tensor, self.weights) + self.bias
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

        dz = output_gradient * self._ACTIVATION_MAP[self.act_func].derivative(self.unactivated_output_cache)
        dw = np.dot(self.input_cache.T, dz) / batch_size
        db = np.sum(dz, axis=0) / batch_size
        di = np.dot(dz, self.weights.T)

        weight_change, bias_change = self.optimizer.adjust_gradient(dw, db, lr)
        self.weights -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

        return di
