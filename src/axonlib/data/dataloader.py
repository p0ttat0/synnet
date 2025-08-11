import os
import gzip
import numpy as np


class Data:
    """
    Data container.

    :var training_data: training data
    :var training_labels: training labels
    :var validation_data: validation data
    :var validation_labels: validation labels
    """
    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray, validation_data: np.ndarray,
                 validation_labels: np.ndarray):
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def shuffle(self, target: str):
        """
        Shuffles the data.
        :param target: 'training' or 'validation'
        """
        if target == 'training':
            assert self.training_data.shape[0] == self.training_labels.shape[0]
            p = np.random.permutation(self.training_data.shape[0])
            self.training_data = self.training_data[p]
            self.training_labels = self.training_labels[p]
        if target == 'validation':
            assert self.validation_data.shape[0] == self.validation_labels.shape[0]
            p = np.random.permutation(self.validation_data.shape[0])
            self.validation_data = self.validation_data[p]
            self.validation_labels = self.validation_labels[p]


class DataLoader:
    """
    Loads data.
    """
    @staticmethod
    def mnist() -> Data:
        """
        Loads mnist and returns training data, training labels, testing data, testing labels
        :return: Data object
        """
        directory = os.path.dirname(__file__)
        training_data_path = os.path.join(directory, r"mnist/train-images-idx3-ubyte.gz")
        training_labels_path = os.path.join(directory, r"mnist/train-labels-idx1-ubyte.gz")
        test_data_path = os.path.join(directory, r"mnist/t10k-images-idx3-ubyte.gz")
        test_label_path = os.path.join(directory, r"mnist/t10k-labels-idx1-ubyte.gz")

        f = gzip.open(training_data_path, 'r')
        f.read(16)
        img_data = np.frombuffer(f.read(60000 * 784), dtype=np.uint8).reshape((60000, 784)) / 255 * 2 - 1
        f = gzip.open(training_labels_path, 'r')
        f.read(8)
        img_labels = np.frombuffer(f.read(60000), dtype=np.uint8).astype(int)[:, np.newaxis]
        img_labels = np.eye(10)[img_labels].squeeze()

        f = gzip.open(test_data_path, 'r')
        f.read(16)
        testing_data = np.frombuffer(f.read(10000 * 784), dtype=np.uint8).reshape((10000, 784)) / 255 * 2 - 1
        f = gzip.open(test_label_path, 'r')
        f.read(8)
        testing_labels = np.frombuffer(f.read(10000), dtype=np.uint8).astype(int)[:, np.newaxis]
        testing_labels = np.eye(10)[testing_labels].squeeze()

        return Data(img_data.astype(np.float32), img_labels, testing_data.astype(np.float32), testing_labels)