from synnet.layers import Dense, Dropout, Flatten, Reshape, Conv, Pool
import synnet as sn
import numpy as np

if __name__ == "__main__":
    m = sn.model.Sequential([
        Reshape((28, 28, 1)),
        Conv((4, 3, 2), padding="valid", stride=1),
        Pool(3, stride=2, padding="valid", pool_mode="average"),
        Flatten(),
        Dense(30, act_func="tanh", weights_init="lecun"),
        Dropout(0.1),
        Dense(20, act_func="relu", weights_init="he"),
        Dense(10, act_func="swish", weights_init="swish"),
        Dense(10, act_func='softmax')
    ])

    m.build((28, 28), "Adam", 'CCE')
    m.forprop(np.random.rand(4, 28, 28))
    m.backprop(np.random.rand(4, 10), 0.01, 0.1)

    data = sn.data.DataLoader.mnist()

    m.fit(data, epochs=1, batch_size=300, learning_rate=0.001, clip_value=1)

