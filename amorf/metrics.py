import numpy as np
from numpy import mean, sqrt
import torch as torch


def average_correlation_coefficient(y_pred, y_true):
    """Calculate Average Correlation Coefficient

    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float: Average Relative Mean Squared Error 

    Raises: 
        ValueError : If Parameters are not both of type np.ndarray or torch.Tensor 
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        top = np.sum((y_true - mean(y_true, axis=0)) *
                     (y_pred - mean(y_pred, axis=0)), axis=0)
        bottom = np.sqrt(np.sum((y_true - mean(y_true, axis=0))**2, axis=0)
                         * np.sum((y_pred - mean(y_pred, axis=0))**2, axis=0))
        return np.sum(top / bottom) / len(y_true[0])
    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        top = torch.sum((y_true - torch.mean(y_true, dim=0))
                        * (y_pred - torch.mean(y_pred, dim=0)), dim=0)
        bottom = torch.sqrt(torch.sum((y_true - torch.mean(y_true, dim=0))**2, dim=0) *
                            torch.sum((y_pred - torch.mean(y_pred, dim=0))**2, dim=0))
        return torch.sum(top / bottom) / len(y_true[0])
    else:
        raise ValueError(
            'y_true and y_pred must be both of type numpy.ndarray or torch.Tensor')


def average_relative_error(y_pred, y_true):
    """Calculate Average Relative Error

    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float: Average Relative Mean Squared Error

    Raises: 
        ValueError : If Parameters are not both of type np.ndarray or torch.Tensor
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sum(abs(y_true - y_pred) / y_true) /
                   len(y_true)) / len(y_true[0, :])
    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sum(torch.abs(y_true - y_pred) / y_true, dim=0) /
                         len(y_true)) / len(y_true[0, :])
    else:
        raise ValueError(
            'y_true and y_pred must be both of type numpy.ndarray or torch.Tensor')


def average_relative_root_mean_squared_error(y_pred, y_true):
    """Calculate Average Relative Root Mean Squared Error (aRRMSE)

    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float : Average Relative Root Mean Squared Error

    Raises: 
        ValueError : If Parameters are not both of type np.ndarray or torch.Tensor
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sqrt(sum((y_true - y_pred)**2) /
                        sum((y_true - mean(y_true, axis=0))**2))) / len(y_pred[0, :])
    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sqrt(torch.sum((y_true - y_pred)**2, dim=0) /
                                    torch.sum(((y_true - torch.mean(y_true, dim=0))**2), dim=0))) / len(y_pred[0, :])
    else:
        raise ValueError(
            'y_true and y_pred must be both of type numpy.ndarray or torch.Tensor')


def mean_squared_error(y_pred, y_true):
    """Calculate Mean Squared Error (MSE)

    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float : Mean Squared Error

    Raises: 
        ValueError : If Parameters are not both of type np.ndarray or torch.Tensor
     """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum((sum((y_true - y_pred)**2) /
                    len(y_true)))
    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sum((y_true - y_pred)**2) /
                         len(y_true))
    else:
        raise ValueError(
            'y_true and y_pred must be both of type numpy.ndarray or torch.Tensor')


def average_root_mean_squared_error(y_pred, y_true):
    """Calculate Average Root Mean Squared Error (aRMSE)

    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Returns:
        float : Average Root Mean Squared Error 

    Raises:
        ValueError : If Parameters are not both of type np.ndarray or torch.Tensor
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        return sum(sqrt((sum((y_true - y_pred)**2) /
                         len(y_true))))/len(y_true[0, :])
    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.sum(torch.sqrt(torch.sum((y_true - y_pred)**2, dim=0)
                                    / len(y_true) ))/len(y_true[0])
    else:
        raise ValueError(
            'y_true and y_pred must be both of type numpy.ndarray or torch.Tensor')


def __validate_dimensions(y_pred, y_true):
    """Validates dimensions of the two input parameters    
    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values

    Raises:
        ValueError: If dimensions are not identical
    """
    if len(y_true) is not len(y_pred) and len(y_true[0]) is not len(y_pred[0]):
        raise ValueError('Dimensions of y_true and y_pred do not match.')
