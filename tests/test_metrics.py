import unittest
import amorf.metrics as metrics
import numpy as np
import torch as torch


class TestError(unittest.TestCase):

    def test_average_correlation_coefficient(self):
        labels = np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])
        predicted_labels = np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])
        self.assertAlmostEqual(metrics.average_correlation_coefficient(
            predicted_labels, labels), -0.725679, places=5)  
    def test_tensor_average_correlation_coefficient(self):
        labels = torch.from_numpy(
            np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])).float()
        predicted_labels = torch.from_numpy(
            np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])).float()
        self.assertAlmostEqual(metrics.average_correlation_coefficient(
            predicted_labels, labels).item(), -0.725679, places=5)

    def test_average_relative_error(self):
        labels = np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])
        predicted_labels = np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])
        self.assertAlmostEqual(metrics.average_relative_error(
            predicted_labels, labels), 0.133333, places=5)
    
    def test_tensor_average_relative_error(self):
        labels = torch.from_numpy(
            np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])).float()
        predicted_labels = torch.from_numpy(
            np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])).float()
        self.assertAlmostEqual(metrics.average_relative_error(
            predicted_labels, labels).item(), 0.133333, places=5)

    def test_average_relative_root_mean_squared_error(self):
        labels = np.array(
            [[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]]).astype(np.float)
        predicted_labels = np.array(
            [[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]]).astype(np.float)
        self.assertAlmostEqual(metrics.average_relative_root_mean_squared_error(
            predicted_labels, labels), 1.8735961929670222, places=5)

    def test_tensor_average_relative_root_mean_squared_error(self):
        labels = torch.from_numpy(
            np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])).float()
        predicted_labels = torch.from_numpy(
            np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])).float()
        self.assertAlmostEqual(metrics.average_relative_root_mean_squared_error(
            predicted_labels, labels), 1.8735961929670222, places=5)

    def test_tensor_mean_squared_error(self):
        labels = torch.from_numpy(
            np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])).float()
        predicted_labels = torch.from_numpy(
            np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])).float()
        self.assertEqual(metrics.mean_squared_error(
            predicted_labels, labels), 108.4)

    def test_mean_squared_error(self): 
        labels = np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])
        predicted_labels = np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])
        self.assertEqual(metrics.mean_squared_error(
            predicted_labels, labels), 108.4) 
        
    def test_tensor_average_root_mean_squared_error(self):
        labels = torch.from_numpy(
            np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])).float()
        predicted_labels = torch.from_numpy(
            np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])).float()
        self.assertAlmostEqual(metrics.average_root_mean_squared_error(
            predicted_labels, labels).item(), 7.36206, places=3) 
    
    def test_average_root_mean_squared_error(self): 
        labels = np.array([[2, -6], [3, 5], [4, 4], [5, 3], [-6, 2]])
        predicted_labels = np.array([[3, 7], [2, -5], [4, 4], [-5, 2], [7, 3]])
        self.assertAlmostEqual(metrics.average_root_mean_squared_error(
            predicted_labels, labels), 7.36206, places=3) 

if __name__ == '__main__':
    unittest.main()
