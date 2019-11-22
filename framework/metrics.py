import numpy as np
from numpy import mean, sqrt
import torch as torch


def average_relative_error(y_test, y_pred):
    """Calculate Average Relative Error

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.array): array of dimension N x d with predicted values

    Returns:
        float: Average Relative Mean Squared Error
    """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sum(abs(y_test - y_pred) / y_test) /
                   len(y_test)) / len(y_test[0, :])
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sum(torch.abs(y_test - y_pred) / y_test, dim=0) /
                       len(y_test)) / len(y_test[0, :])
    else: 
        raise ValueError('y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')

def average_relative_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Relative Root Mean Squared Error (aRRMSE)

    Args:
        y_test (np.array): array of dimension N x d with actual values
        y_pred (np.array): array of dimension N x d with predicted values

    Returns:
        float : Average Relative Root Mean Squared Error
    """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sqrt(sum((y_test - y_pred)**2) /
                        sum((y_test - mean(y_test, axis=0))**2))) / len(y_pred[0, :]) 
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sqrt(torch.sum((y_test - y_pred)**2, dim=0) /
                                  torch.sum(((y_test - torch.mean(y_test, dim=0))**2), dim=0))) / len(y_pred[0, :])
    else: 
        raise ValueError('y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')

def mean_squared_error(y_pred, y_test):
    """Calculate Mean Squared Error (MSE)

     Args:
         y_test (np.array): array of dimension N x d with actual values
         y_pred (np.ar ray): array of dimension N x d with predicted values

     Returns:
         float : Mean Squared Error
     """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum((sum((y_test - y_pred)**2) /
                    len(y_test))) 
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sum((y_test - y_pred)**2) /
                       len(y_test))
    else: 
        raise ValueError('y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')

def average_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Root Mean Squared Error (aRMSE)

     Args:
         y_test (np.array): array of dimension N x d with actual values
         y_pred (np.ar ray): array of dimension N x d with predicted values

    Returns:
        float : Average Root Mean Squared Error
     """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sqrt((sum((y_test - y_pred)**2)
                       / len(y_test))  ))/len(y_test[0, :]) 
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sqrt(torch.sum((y_test - y_pred)**2, dim=0)
                                  / len(y_test)  ))/len(y_test[0])
    else: 
        raise ValueError('y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')
