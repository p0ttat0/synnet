from typing import Tuple, List
from synnet.base.learnable_layer_base import LearnableLayerBase
from synnet.base.utility_layer_base import UtilityLayerBase
from synnet.data.dataloader import Data
from synnet.layers.dropout import Dropout
from synnet.model.loss_functions import CCE, MSE
from synnet.optimizers.adam import Adam
from synnet.optimizers.no_optimizer import NoOptimizer
import numpy as np
import onnx
from synnet.model.helpers.training_display import ProgressBar
from synnet.layers.reshape import Reshape
from synnet.layers.dropout import Dropout
from synnet.layers.pool import Pool
from synnet.layers.convolution import Conv
from synnet.layers.flatten import Flatten
from synnet.layers.dense import Dense


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
            'accuracy': lambda training_predictions, training_labels: np.sum(np.argmax(training_predictions, axis=1) == np.argmax(training_labels, axis=1)) / training_labels.shape[0],
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
            if not isinstance(layer, Dropout) or not training:
                input_tensor = layer.forprop(input_tensor)
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
            if isinstance(layer, LearnableLayerBase):
                output_gradient = layer.backprop(output_gradient, lr, clip_value)
            else:
                output_gradient = layer.backprop(output_gradient)

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

    def save_onnx(self, file_path: str, input_shape: Tuple[int, ...]):
        if not self.initialized:
            raise Exception("model not initialized")

        nodes = []
        initializers = []
        input_name = "input"
        layer_input = input_name

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                # Weights
                weight_name = f"W_{i}"
                initializers.append(
                    onnx.helper.make_tensor(
                        name=weight_name,
                        data_type=self._np_to_onnx_type(layer.weights.dtype),
                        dims=layer.weights.shape,
                        vals=layer.weights.flatten().tolist(),
                    )
                )

                # MatMul
                matmul_output = f"matmul_output_{i}"
                nodes.append(
                    onnx.helper.make_node(
                        "MatMul",
                        [layer_input, weight_name],
                        [matmul_output],
                        name=f"MatMul_{i}",
                    )
                )

                # Bias
                bias_name = f"B_{i}"
                initializers.append(
                    onnx.helper.make_tensor(
                        name=bias_name,
                        data_type=self._np_to_onnx_type(layer.bias.dtype),
                        dims=layer.bias.shape,
                        vals=layer.bias.flatten().tolist(),
                    )
                )

                # Add
                add_output = f"add_output_{i}"
                nodes.append(
                    onnx.helper.make_node(
                        "Add",
                        [matmul_output, bias_name],
                        [add_output],
                        name=f"Add_{i}",
                    )
                )

                # Activation
                activation_output = f"activation_output_{i}"
                nodes.append(
                    onnx.helper.make_node(
                        layer.act_func.capitalize(),
                        [add_output],
                        [activation_output],
                        name=f"Activation_{i}",
                    )
                )
                layer_input = activation_output
            else:
                nodes, initializers, layer_input = self._layer_to_onnx(layer, i, layer_input, initializers)

        # Create the graph
        graph_def = onnx.helper.make_graph(
            nodes,
            "synnet-model",
            [onnx.helper.make_tensor_value_info(input_name, self._np_to_onnx_type(np.float32), [None, *input_shape])],
            [onnx.helper.make_tensor_value_info(layer_input, self._np_to_onnx_type(np.float32), [None, self.layers[-1].size])],
            initializer=initializers,
        )

        # Create the model
        model_def = onnx.helper.make_model(graph_def, producer_name="synnet")
        onnx.save(model_def, file_path)

    def _layer_to_onnx(self, layer, i, layer_input, initializers):
        nodes = []
        if isinstance(layer, Flatten):
            flatten_output = f"flatten_output_{i}"
            nodes.append(
                onnx.helper.make_node(
                    "Flatten",
                    [layer_input],
                    [flatten_output],
                    name=f"Flatten_{i}",
                )
            )
            layer_input = flatten_output
        elif isinstance(layer, Conv):
            # Weights
            weight_name = f"W_{i}"
            initializers.append(
                onnx.helper.make_tensor(
                    name=weight_name,
                    data_type=self._np_to_onnx_type(layer.weights.dtype),
                    dims=layer.weights.shape,
                    vals=layer.weights.flatten().tolist(),
                )
            )

            # Conv
            conv_output = f"conv_output_{i}"
            nodes.append(
                onnx.helper.make_node(
                    "Conv",
                    [layer_input, weight_name],
                    [conv_output],
                    name=f"Conv_{i}",
                    strides=layer.stride,
                    pads=[layer.padding[0], layer.padding[1], layer.padding[0], layer.padding[1]],
                )
            )

            # Bias
            bias_name = f"B_{i}"
            initializers.append(
                onnx.helper.make_tensor(
                    name=bias_name,
                    data_type=self._np_to_onnx_type(layer.bias.dtype),
                    dims=(1, layer.bias.shape[0], 1, 1),
                    vals=layer.bias.flatten().tolist(),
                )
            )

            # Add
            add_output = f"add_output_{i}"
            nodes.append(
                onnx.helper.make_node(
                    "Add",
                    [conv_output, bias_name],
                    [add_output],
                    name=f"Add_{i}",
                )
            )

            # Activation
            activation_output = f"activation_output_{i}"
            nodes.append(
                onnx.helper.make_node(
                    layer.act_func.capitalize(),
                    [add_output],
                    [activation_output],
                    name=f"Activation_{i}",
                )
            )
            layer_input = activation_output
        elif isinstance(layer, Pool):
            pool_output = f"pool_output_{i}"
            pool_type = "MaxPool" if layer.pool_mode == "max" else "AveragePool"
            nodes.append(
                onnx.helper.make_node(
                    pool_type,
                    [layer_input],
                    [pool_output],
                    name=f"Pool_{i}",
                    kernel_shape=[layer.kernel_size, layer.kernel_size],
                    strides=layer.stride,
                    pads=[layer.padding[0], layer.padding[1], layer.padding[0], layer.padding[1]],
                )
            )
            layer_input = pool_output
        elif isinstance(layer, Dropout):
            dropout_output = f"dropout_output_{i}"
            nodes.append(
                onnx.helper.make_node(
                    "Dropout",
                    [layer_input],
                    [dropout_output],
                    name=f"Dropout_{i}",
                    ratio=layer.dropout_rate,
                )
            )
            layer_input = dropout_output
        elif isinstance(layer, Reshape):
            reshape_output = f"reshape_output_{i}"
            shape_name = f"shape_{i}"
            initializers.append(
                onnx.helper.make_tensor(
                    name=shape_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=(len(layer.output_shape) + 1,),
                    vals=[-1] + list(layer.output_shape),
                )
            )
            nodes.append(
                onnx.helper.make_node(
                    "Reshape",
                    [layer_input, shape_name],
                    [reshape_output],
                    name=f"Reshape_{i}",
                )
            )
            layer_input = reshape_output
        else:
            print(f"Skipping layer of type {type(layer)} as it is not supported for ONNX export.")
        return nodes, initializers, layer_input

    @staticmethod
    def load_onnx(file_path: str):
        model = onnx.load(file_path)
        layers = []
        # This is a simplified loader and will not work for complex models
        # It assumes a simple sequential model with Dense layers
        for node in model.graph.node:
            if node.op_type == "Flatten":
                layers.append(Flatten())
            if node.op_type == "Conv":
                weights = None
                bias = None
                for initializer in model.graph.initializer:
                    if initializer.name == node.input[1]:
                        weights = onnx.numpy_helper.to_array(initializer)

                add_node = next((n for n in model.graph.node if n.op_type == "Add" and n.input[0] == node.output[0]), None)
                if add_node:
                    for initializer in model.graph.initializer:
                        if initializer.name == add_node.input[1]:
                            bias = onnx.numpy_helper.to_array(initializer)

                activation_node = next((n for n in model.graph.node if n.input[0] == add_node.output[0]), None)

                if weights is not None and bias is not None and activation_node:
                    layer = Conv(
                        kernel_params=(weights.shape[0], weights.shape[1], weights.shape[3]),
                        act_func=activation_node.op_type.lower(),
                        padding='valid' if node.pads == [0, 0, 0, 0] else 'full',
                        stride=node.strides,
                    )
                    layer.weights = weights
                    layer.bias = bias.flatten()
                    layers.append(layer)
            if node.op_type in ["MaxPool", "AveragePool"]:
                layer = Pool(
                    kernel_size=node.attribute[0].ints[0],
                    stride=node.attribute[2].ints,
                    padding='valid' if node.attribute[1].ints == [0, 0, 0, 0] else 'full',
                    pool_mode="max" if node.op_type == "MaxPool" else "average",
                )
                layers.append(layer)
            if node.op_type == "Dropout":
                layers.append(Dropout(dropout_rate=node.attribute[0].f))
            if node.op_type == "Reshape":
                shape_tensor = next(t for t in model.graph.initializer if t.name == node.input[1])
                shape = onnx.numpy_helper.to_array(shape_tensor)
                layers.append(Reshape(output_shape=tuple(shape[1:])))
            if node.op_type == "MatMul":
                weights = None
                bias = None
                for initializer in model.graph.initializer:
                    if initializer.name == node.input[1]:
                        weights = onnx.numpy_helper.to_array(initializer)
                
                # Find the corresponding to Add node for the bias
                add_node = next((n for n in model.graph.node if n.op_type == "Add" and n.input[0] == node.output[0]), None)
                if add_node:
                    for initializer in model.graph.initializer:
                        if initializer.name == add_node.input[1]:
                            bias = onnx.numpy_helper.to_array(initializer)

                # Find the corresponding Activation node
                activation_node = next((n for n in model.graph.node if n.input[0] == add_node.output[0]), None)
                
                if weights is not None and bias is not None and activation_node:
                    layer = Dense(size=weights.shape[1], act_func=activation_node.op_type.lower())
                    layer.weights = weights
                    layer.bias = bias
                    layers.append(layer)

        return Sequential(layers)

    @staticmethod
    def _np_to_onnx_type(np_type):
        if np_type == np.float32:
            return onnx.TensorProto.FLOAT
        elif np_type == np.float64:
            return onnx.TensorProto.DOUBLE
        elif np_type == np.int64:
            return onnx.TensorProto.INT64
        elif np_type == np.int32:
            return onnx.TensorProto.INT32
        else:
            raise Exception(f"Unsupported numpy type: {np_type}")
