import time
from typing import Tuple, List
from synnet.base.learnable_layer_base import LearnableLayerBase
from synnet.base.utility_layer_base import UtilityLayerBase
from synnet.data.dataloader import Data
from synnet.layers.dropout import Dropout
from synnet.model.loss_functions import CCE, MSE
from synnet.optimizers.adam import Adam
from synnet.optimizers.no_optimizer import NoOptimizer
import numpy as np
from synnet.layers.helpers.training_display import ProgressBar


class Sequential:
    """
    Sequential model.

    :var layers: list of layers
    :var initialized: whether the model is initialized
    :var optimizer: optimizer
    :var loss_func: loss function
    """
    def __init__(self, layers: List[UtilityLayerBase | LearnableLayerBase] = None):
        self._LOSS_MAP = {
            'CCE': CCE(),
            'MSE': MSE()
        }
        self._METRICS_MAP = {
            'accuracy': lambda x, y: 0,
            'precision': lambda x, y: 0,
            'recall': lambda x, y: 0
        }

        self.layers = [] if layers is None else layers
        self.initialized = False
        self.optimizer = None
        self.loss_func = None

    def build(self, input_shape: Tuple[int, ...], optimizer= 'Adam', loss_func= 'CCE'):
        """
        Initializes the model.
        :param input_shape: input shape
        :param optimizer: optimizer
        :param loss_func: loss function
        """
        if optimizer not in ['Adam', 'none', 'no optimizer']: raise Exception(f"dense layer '{optimizer}' optimizer is not supported")
        if loss_func not in ['CCE', 'MSE']: raise Exception(f"dense layer '{loss_func}' loss function is not supported")

        self.optimizer = optimizer
        self.loss_func = loss_func

        for layer in self.layers:
            input_shape = layer.param_init(input_shape)
            if isinstance(layer, LearnableLayerBase):
                match optimizer:
                    case 'Adam':
                        optimizer_obj = Adam()
                        optimizer_obj.fme["weights"] = np.zeros(layer.weights.shape)
                        optimizer_obj.fme["bias"] = np.zeros(layer.bias.shape)
                        optimizer_obj.sme["weights"] = np.zeros(layer.weights.shape)
                        optimizer_obj.sme["bias"] = np.zeros(layer.bias.shape)
                    case "no optimizer" | "none":
                        optimizer_obj = NoOptimizer()
                    case _:
                        raise NotImplementedError(f"{optimizer} optimizer not implemented")
                layer.optimizer = optimizer_obj

        self.initialized = True
        self.loss_func = loss_func

    def forprop(self, input_tensor: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward propagation through all layers.
        :param input_tensor: input tensor
        :param training: whether the model is training
        :return: model output
        """
        if not self.initialized:
            raise Exception("model not initialized")

        for layer in self.layers:
            # s_time = time.time()
            if not isinstance(layer, Dropout) or not training:
                input_tensor = layer.forprop(input_tensor)
            # print(f"type: {type(layer)} | {time.time() - s_time}s")
        return input_tensor

    def backprop(self, output_gradient: np.ndarray, lr: float, clip_value: float):
        """
        Backward propagation through all layers.
        :param output_gradient: output gradient
        :param lr: learning rate
        :param clip_value: clip value
        """
        if not self.initialized:
            raise Exception("model not initialized")

        for layer in reversed(self.layers):
            # s_time = time.time()
            if isinstance(layer, LearnableLayerBase):
                output_gradient = layer.backprop(output_gradient, lr, clip_value)
            else:
                output_gradient = layer.backprop(output_gradient)
            # print(f"type: {type(layer)} | {time.time() - s_time}s")

    def fit(self, data: Data, epochs: int, batch_size: int, learning_rate: float, clip_value: float, metrics: List[str] = None):
        """
        Fits the model to the data.
        :param data: data to fit to
        :param epochs: number of epochs
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param clip_value: clip value
        :param metrics: list of metrics to track
        """
        if not self.initialized:
            raise Exception("model not initialized")

        if metrics is None:
            metrics = []

        for metric in metrics:
            if metric not in self._METRICS_MAP:
                raise ValueError(f"Metric '{metric}' not supported.")

        batches_per_epoch = data.training_data.shape[0] // batch_size
        progress_bar = ProgressBar(total_steps=batches_per_epoch, epochs=epochs)
        progress_bar.start()

        for epoch in range(epochs):
            data.shuffle('training')
            for batch in range(batches_per_epoch):
                training_labels = data.training_labels[batch * batch_size:(batch + 1) * batch_size]
                training_data = data.training_data[batch * batch_size:(batch + 1) * batch_size]
                training_predictions = self.forprop(training_data, training=True)

                loss = self._LOSS_MAP[self.loss_func].get_loss(training_labels, training_predictions, batch_size)
                d_loss = self._LOSS_MAP[self.loss_func].get_d_loss(training_labels, training_predictions)

                self.backprop(d_loss, learning_rate, clip_value)

                metric_results = {'loss': loss}
                for metric in metrics:
                    metric_results[metric] = self._METRICS_MAP[metric](training_labels, training_predictions)

                progress_bar.update(epoch, batch, metric_results)
        progress_bar.end()
