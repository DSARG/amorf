import numpy as np
import arff


class EDM():
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
            tuple: Two Numpy Array X and y of type np.float32

        """
        return self.X, self.y

    def get_pytorch_dataloader(self):
        """Retrieve X and Y as PyTorch dataLoaders

        Returns dataloaders with two arrays, X and y

        Args:

        Returns:
            tuple: Two PyTorch dataloaders X and y

        """
        raise NotImplementedError
