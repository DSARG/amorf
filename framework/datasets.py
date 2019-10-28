import arff
import numpy as np


class EDM():
    def __init__(self):
        with open('data/edm.arff') as file:
            dataset = arff.load(file)
            data = np.array(dataset['data'])
            self.X = X = data[:, 0:16].astype(np.float32)
            self.y = y = data[:, 16:18].astype(np.float32)
            self.X.astype(np.float32)
            self.y.astype(np.float32)

    def get_numpy(self):
        """Retrieve X and Y as Numpy Array

        Returns tuple with two arrays, X and y

        Args:

        Returns:
            (np.ndarray, np.ndarray): Numpy Arraya X and y of type np.float32

        """
        return self.X, self.y

    def get_pytorch_dataloader(self):
        """Retrieve X and Y as PyTorch dataLoaders

        Returns dataloaders with two arrays, X and y

        Args:

        Returns:
            (np.ndarray, np.ndarray): Two PyTorch dataloaders X and y

        """
        raise NotImplementedError


class RiverFlow1():
    def __init__(self):
        with open('data/rf1.arff') as file:
            dataset = arff.load(file)
            data = np.array(dataset['data'])
            X = data[:, 0:64]
            y = data[:, 64:72]
            X[X == None] = 0
            y[y == None] = 0
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)

    def get_numpy(self):
        """Retrieve X and Y as Numpy Array

        Returns tuple with two arrays, X and y
        Replaces NaN values with 0

        Args:

        Returns:
            (np.ndarray, np.ndarray): Numpy Arraya X and y of type np.float32

        """
        return self.X, self.y

    def get_pytorch_dataloader(self):
        """Retrieve X and Y as PyTorch dataLoaders

        Returns dataloaders with two arrays, X and y

        Args:

        Returns:
            (np.ndarray, np.ndarray): Two PyTorch dataloaders X and y

        """
        raise NotImplementedError


class WaterQuality():
    def __init__(self):
        with open('data/wq.arff') as file:
            dataset = arff.load(file)
            data = np.array(dataset['data'])
            self.X = data[:, 16:30].astype(np.float32)
            self.y = data[:, 0:16].astype(np.float32)

    def get_numpy(self):
        """Retrieve X and Y as Numpy Array

        Returns tuple with two arrays, X and y

        Args:

        Returns:
            (np.ndarray, np.ndarray): Numpy Arraya X and y of type np.float32

        """
        return self.X, self.y

    def get_pytorch_dataloader(self):
        """Retrieve X and Y as PyTorch dataLoaders

        Returns dataloaders with two arrays, X and y

        Args:

        Returns:
            (np.ndarray, np.ndarray): Two PyTorch dataloaders X and y

        """
        raise NotImplementedError


class TransparentConductors():
    def __init__(self):
        data = np.loadtxt('data/transparentConductors.csv',
                             delimiter=",", skiprows=1)
        self.X = data[:, 0:12].astype(np.float32)
        self.y = data[:, 12:14].astype(np.float32)

    def get_numpy(self):
        """Retrieve X and Y as Numpy Array

        Returns tuple with two arrays, X and y

        Args:

        Returns:
            (np.ndarray, np.ndarray): Numpy Arraya X and y of type np.float32

        """
        return self.X, self.y

    def get_pytorch_dataloader(self):
        """Retrieve X and Y as PyTorch dataLoaders

        Returns dataloaders with two arrays, X and y

        Args:

        Returns:
            (np.ndarray, np.ndarray): Two PyTorch dataloaders X and y

        """
        return NotImplementedError