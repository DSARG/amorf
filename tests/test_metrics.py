import unittest
import framework.metrics as metrics
import numpy as np
import torch as torch


class TestError(unittest.TestCase):

    def test_average_relative_error(self):
        labels = np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])
        predicted_labels = np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])
        self.assertAlmostEqual(metrics.average_relative_error(
            labels, predicted_labels), 0.133333, places=5)

    def test_average_relative_root_mean_squared_error(self):
        labels = np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]]).astype(np.float)
        predicted_labels = np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]]).astype(np.float)
        self.assertAlmostEqual(metrics.average_relative_root_mean_squared_error(
            labels, predicted_labels), 1.873596193, places=5)

    def test_tensor_average_relative_root_mean_squared_error(self):
        labels = torch.from_numpy(np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])).float()
        predicted_labels = torch.from_numpy(np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])).float()
        self.assertAlmostEqual(metrics.tensor_average_relative_root_mean_squared_error(
            labels, predicted_labels).item(), 1.873596193, places=5)

if __name__ == '__main__':
    unittest.main()
