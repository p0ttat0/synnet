import os
import gzip
import numpy as np


class Data:
    """
    A container for training and validation data.

    Attributes:
        training_data (np.ndarray): The training data.
        training_labels (np.ndarray): The labels for the training data.
        validation_data (np.ndarray): The validation data.
        validation_labels (np.ndarray): The labels for the validation data.
    """

    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray, validation_data: np.ndarray,
                 validation_labels: np.ndarray):
        """
        Initializes the Data object.

        Args:
            training_data: The training data.
            training_labels: The labels for the training data.
            validation_data: The validation data.
            validation_labels: The labels for the validation data.
        """
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def shuffle(self, target: str):
        """
        Shuffles the specified dataset.

        Args:
            target: The dataset to shuffle, either 'training' or 'validation'.
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
    A class for loading datasets.
    """

    @staticmethod
    def mnist() -> Data:
        """
        Loads the MNIST dataset.

        Returns:
            A Data object containing the MNIST training and validation data.
        """
        directory = os.path.dirname(__file__)
        training_data_path = os.path.join(directory, r"mnist/train-images-idx3-ubyte.gz")
        training_labels_path = os.path.join(directory, r"mnist/train-labels-idx1-ubyte.gz")
        test_data_path = os.path.join(directory, r"mnist/t10k-images-idx3-ubyte.gz")
        test_label_path = os.path.join(directory, r"mnist/t10k-labels-idx1-ubyte.gz")

        with gzip.open(training_data_path, 'r') as f:
            f.read(16)
            img_data = np.frombuffer(f.read(60000 * 784), dtype=np.uint8).reshape((60000, 784)) / 255 * 2 - 1
        with gzip.open(training_labels_path, 'r') as f:
            f.read(8)
            img_labels = np.frombuffer(f.read(60000), dtype=np.uint8).astype(int)[:, np.newaxis]
            img_labels = np.eye(10)[img_labels].squeeze()

        with gzip.open(test_data_path, 'r') as f:
            f.read(16)
            testing_data = np.frombuffer(f.read(10000 * 784), dtype=np.uint8).reshape((10000, 784)) / 255 * 2 - 1
        with gzip.open(test_label_path, 'r') as f:
            f.read(8)
            testing_labels = np.frombuffer(f.read(10000), dtype=np.uint8).astype(int)[:, np.newaxis]
            testing_labels = np.eye(10)[testing_labels].squeeze()

        return Data(img_data.astype(np.float32), img_labels, testing_data.astype(np.float32), testing_labels)
