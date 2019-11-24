import numpy as np
from numpy import mean, sqrt
import torch as torch

# TODO: add function to check diemsnions


def average_correlation_coefficient(y_test, y_pred):
    """ 

    Args:
        y_test ([type]): [description]
        y_pred ([type]): [description]
    """
    #mean_pred = mean(y_pred, axis=0)
    #mean_test = mean(y_test, axis=0)
    #l = (y_test - mean(y_test, axis=0))
    #r= (y_pred - mean(y_pred, axis=0))
    #o = l*r
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        top = np.sum((y_test - mean(y_test, axis=0))
                     * (y_pred - mean(y_pred, axis=0)), axis=0)
        bottom = np.sqrt(np.sum((y_test - mean(y_test, axis=0))**2, axis=0) *
                         np.sum((y_pred - mean(y_pred, axis=0))**2, axis=0))
        return np.sum(top / bottom) / len(y_test[0])
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor): 
        top = torch.sum((y_test - torch.mean(y_test, dim=0)) *
                        (y_pred - torch.mean(y_pred, dim=0)), dim=0)
        bottom = torch.sqrt(torch.sum((y_test - torch.mean(y_test, dim=0))**2, dim=0)
                            * torch.sum((y_pred - torch.mean(y_pred, dim=0))**2, dim=0))
        return torch.sum(top / bottom) / len(y_test[0])
    else:
        raise ValueError(
            'y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')


def average_relative_error(y_test, y_pred):
    """Calculate Average Relative Error

    Args:
        y_test (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float: Average Relative Mean Squared Error
    """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sum(abs(y_test - y_pred) / y_test)
                   / len(y_test)) / len(y_test[0, :])
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sum(torch.abs(y_test - y_pred) / y_test, dim=0)
                         / len(y_test)) / len(y_test[0, :])
    else:
        raise ValueError(
            'y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')


def average_relative_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Relative Root Mean Squared Error (aRRMSE)

    Args:
        y_test (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float : Average Relative Root Mean Squared Error
    """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sqrt(sum((y_test - y_pred)**2)
                        / sum((y_test - mean(y_test, axis=0))**2))) / len(y_pred[0, :])
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sqrt(torch.sum((y_test - y_pred)**2, dim=0)
                                    / torch.sum(((y_test - torch.mean(y_test, dim=0))**2), dim=0))) / len(y_pred[0, :])
    else:
        raise ValueError(
            'y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')


def mean_squared_error(y_pred, y_test):
    """Calculate Mean Squared Error (MSE)

     Args:
         y_test (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
         y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

     Returns:
         float : Mean Squared Error
     """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum((sum((y_test - y_pred)**2)
                    / len(y_test)))
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sum((y_test - y_pred)**2)
                         / len(y_test))
    else:
        raise ValueError(
            'y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')


def average_root_mean_squared_error(y_pred, y_test):
    """Calculate Average Root Mean Squared Error (aRMSE)

     Args:
         y_test (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
         y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float : Average Root Mean Squared Error
     """
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sqrt((sum((y_test - y_pred)**2)
                         / len(y_test))   ))/len(y_test[0, :])
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sqrt(torch.sum((y_test - y_pred)**2, dim=0) /
                                    len(y_test)   ))/len(y_test[0])
    else:
        raise ValueError(
            'y_test and y_pred must be both of type numpy.ndarray or torch.Tensor')
