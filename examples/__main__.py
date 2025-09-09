from synnet.layers import Dense, Dropout, Flatten, Reshape, Conv, Pool
import synnet as sn
import numpy as np

if __name__ == "__main__":
    m = sn.model.Sequential([
        Reshape((28, 28, 1)),
        Conv((3, 3, 4), padding="valid", stride=2),
        Dropout(0.1),
        Pool(3, stride=2, padding="full", pool_mode="max"),
        Flatten(),
        Dense(128, act_func="swish", weights_init="swish"),
        Dropout(0.1),
        Dense(32, act_func="swish", weights_init="swish"),
        Dense(24, act_func="swish", weights_init="swish"),
        Dense(10, act_func='softmax')
    ])

    m.build((28, 28), "Adam", 'CCE')
    m.forprop(np.random.rand(4, 28, 28))
    m.backprop(np.random.rand(4, 10), 0.01, 0.1)

    data = sn.data.DataLoader.mnist()

    m.fit(data, epochs=1, batch_size=256, learning_rate=0.001, clip_value=1, metrics=["accuracy"])
    m.save_onnx("/home/ericl/PycharmProjects/synnet/examples/model.onnx", input_shape=(28, 28))
    m.load_onnx("/home/ericl/PycharmProjects/synnet/examples/model.onnx")
    m.fit(data, epochs=1, batch_size=256, learning_rate=0.001, clip_value=1, metrics=["accuracy"])

